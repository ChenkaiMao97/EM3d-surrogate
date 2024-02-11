import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
# from utilities3 import *

# from Adam import Adam

# torch.manual_seed(0)
# np.random.seed(0)

###############
# Loss
###############
#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=1, p=1, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

################################################################
# fourier layer
################################################################
class Modulated_SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(Modulated_SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        
        # single model training (on sherlock)
        #self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        #self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        
        # if for DDP, cfloat is not supported:
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))

        #nn.init.xavier_uniform_(self.weights1)
        #nn.init.xavier_uniform_(self.weights2)

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        # single model training
        #out_ft[:, :, :self.modes1, :self.modes2] = \
        #    self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1*mod1)
        #out_ft[:, :, -self.modes1:, :self.modes2] = \
        #    self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2*mod2)

        # for DDP:
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], torch.view_as_complex(self.weights1))
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], torch.view_as_complex(self.weights2))

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO_multimodal_2d(nn.Module):
    def __init__(self, args):
        super(FNO_multimodal_2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = args.f_modes
        self.modes2 = args.f_modes
        self.width = args.HIDDEN_DIM
        self.padding = args.f_padding # pad the domain if input is non-periodic
        self.sizex = args.domain_sizex
        self.sizey = args.domain_sizey

        self.pre_data_channels = 6
        self.fc0_dielectric = nn.Linear(self.pre_data_channels, self.width) # input channel is 3: (a(x, y), x, y), 
        #-2 is because there are additional 2 channel of boundary fields
        
        # self.fc0_bc = nn.Linear(2, self.width//2)

        # self.fc_bc_1 = nn.Linear(2*(self.sizex+self.sizey), self.sizex*self.sizey)
        # self.fc_bc_2 = nn.Linear(self.sizex*self.sizey    , self.sizex*self.sizey)

        self.num_fourier_layers = args.num_fourier_layers
        self.ALPHA = args.ALPHA

        self.convs = []
        self.ws = []
        for i in range(self.num_fourier_layers):
            self.convs.append(Modulated_SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
            self.ws.append(nn.Conv2d(self.width, self.width, 1))
        self.convs = nn.ModuleList(self.convs)
        self.ws = nn.ModuleList(self.ws)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, args.outc)

        self.loss_fn = LpLoss(size_average=True)

    def forward(self, yeex, yeey, top_bc, bottom_bc, left_bc, right_bc, output_init=False):
        # Sx_f: [bs, subdomain_size, subdomain_size,2]
        # Sy_f: [bs, subdomain_size, subdomain_size,2]
        # source: [bs, subdomain_size, subdomain_size, 2]
        # yeex: [bs, subdomain_size, subdomain_size]
        # yeey: [bs, subdomain_size, subdomain_size]
        # top_bc, bottom_bc: [bs, 1, subdomain_size, 2]
        # left_bc, right_bc: [bs, subdomain_size, 1, 2]

        batch_size = yeex.shape[0]

        preprocessed = torch.zeros((yeex.shape[0], yeex.shape[1], yeex.shape[2], 2), device=yeex.device)

        # just use bc itself and leave 0s in the middle
        for channel in [0,1]:
            preprocessed[:,0,1:-1,channel] = torch.squeeze(top_bc,dim=1)[:,1:-1,channel]
            preprocessed[:,-1,1:-1,channel] = torch.squeeze(bottom_bc,dim=1)[:,1:-1,channel]
            preprocessed[:,1:-1,0,channel] = torch.squeeze(left_bc,dim=2)[:,1:-1,channel]
            preprocessed[:,1:-1,-1,channel] = torch.squeeze(right_bc,dim=2)[:,1:-1,channel]

        grid = self.get_grid(yeex.shape, yeex.device)
        
        pre_data = torch.cat((yeex.unsqueeze(dim=1), yeey.unsqueeze(dim=1), grid, preprocessed[:,:,:,0].unsqueeze(dim=1), preprocessed[:,:,:,1].unsqueeze(dim=1)), dim=1)

        x = pre_data.permute(0, 2, 3, 1)
        x = self.fc0_dielectric(x)
        x = x.permute(0, 3, 1, 2)
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        for i in range(self.num_fourier_layers-1):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            x = F.leaky_relu(x, negative_slope=self.ALPHA)

        x1 = self.convs[-1](x)
        x2 = self.ws[-1](x)
        x = x1 + x2

        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=self.ALPHA)
        x = self.fc2(x)

        if output_init:
            return x, preprocessed
        else:
            return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)

################################################################
# configs
################################################################
# TRAIN_PATH = 'data/piececonst_r421_N1024_smooth1.mat'
# TEST_PATH = 'data/piececonst_r421_N1024_smooth2.mat'

# ntrain = 1000
# ntest = 100

# batch_size = 20
# learning_rate = 0.001

# epochs = 500
# step_size = 100
# gamma = 0.5

# modes = 12
# width = 32

# r = 5
# h = int(((421 - 1)/r) + 1)
# s = h

################################################################
# load data and data normalization
################################################################
# reader = MatReader(TRAIN_PATH)
# x_train = reader.read_field('coeff')[:ntrain,::r,::r][:,:s,:s]
# y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]

# reader.load_file(TEST_PATH)
# x_test = reader.read_field('coeff')[:ntest,::r,::r][:,:s,:s]
# y_test = reader.read_field('sol')[:ntest,::r,::r][:,:s,:s]

# x_normalizer = UnitGaussianNormalizer(x_train)
# x_train = x_normalizer.encode(x_train)
# x_test = x_normalizer.encode(x_test)

# y_normalizer = UnitGaussianNormalizer(y_train)
# y_train = y_normalizer.encode(y_train)

# x_train = x_train.reshape(ntrain,s,s,1)
# x_test = x_test.reshape(ntest,s,s,1)

# train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

# ################################################################
# # training and evaluation
# ################################################################
# model = FNO_multimodal_2d(modes, modes, width).cuda()
# print(count_params(model))

# optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# myloss = LpLoss(size_average=False)
# y_normalizer.cuda()
# for ep in range(epochs):
#     model.train()
#     t1 = default_timer()
#     train_l2 = 0
#     for x, y in train_loader:
#         x, y = x.cuda(), y.cuda()

#         optimizer.zero_grad()
#         out = model(x).reshape(batch_size, s, s)
#         out = y_normalizer.decode(out)
#         y = y_normalizer.decode(y)

#         loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
#         loss.backward()

#         optimizer.step()
#         train_l2 += loss.item()

#     scheduler.step()

#     model.eval()
#     test_l2 = 0.0
#     with torch.no_grad():
#         for x, y in test_loader:
#             x, y = x.cuda(), y.cuda()

#             out = model(x).reshape(batch_size, s, s)
#             out = y_normalizer.decode(out)

#             test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

#     train_l2/= ntrain
#     test_l2 /= ntest

#     t2 = default_timer()
#     print(ep, t2-t1, train_l2, test_l2)
