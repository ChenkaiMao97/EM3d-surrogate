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
# from timeit import default_timer

################################################################
# 3d fourier layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype=torch.float32))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype=torch.float32))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype=torch.float32))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype=torch.float32))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], torch.view_as_complex(self.weights1))
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], torch.view_as_complex(self.weights2))
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], torch.view_as_complex(self.weights3))
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], torch.view_as_complex(self.weights4))

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO_multimodal_3d(nn.Module):
    def __init__(self, args):
        super(FNO_multimodal_3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = args.f_modes_x
        self.modes2 = args.f_modes_y
        self.modes3 = args.f_modes_z
        self.width = args.HIDDEN_DIM
        
        self.padding_x = args.x_padding 
        self.padding_y = args.y_padding 
        self.padding_z = args.z_padding 

        self.input_data_channels = 4
        self.fc0 = nn.Linear(self.input_data_channels, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.num_fourier_layers = args.num_fourier_layers
        self.ALPHA = args.ALPHA

        self.convs = []
        self.ws = []

        for i in range(self.num_fourier_layers):
            self.convs.append(SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3))
            self.ws.append(nn.Conv3d(self.width, self.width, 1))
        self.convs = nn.ModuleList(self.convs)
        self.ws = nn.ModuleList(self.ws)

        # self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        # self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        # self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        # self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        # self.w0 = nn.Conv3d(self.width, self.width, 1)
        # self.w1 = nn.Conv3d(self.width, self.width, 1)
        # self.w2 = nn.Conv3d(self.width, self.width, 1)
        # self.w3 = nn.Conv3d(self.width, self.width, 1)
        # self.bn0 = torch.nn.BatchNorm3d(self.width)
        # self.bn1 = torch.nn.BatchNorm3d(self.width)
        # self.bn2 = torch.nn.BatchNorm3d(self.width)
        # self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, args.outc)

    def forward(self, yeez):
        batch_size = yeez.shape[0] # shape: [bs, sx, sy, sz]
        grid = self.get_grid(yeez.shape, yeez.device) # shape: [bs, sx, sy, sz, 3]

        input_data = torch.cat((yeez[:,:,:,:,None], grid), dim=-1) # shape: [bs, sx, sy, sz, 6]

        x = self.fc0(input_data) # shape: [16, 96, 96, 64, 8]
        x = x.permute(0, 4, 1, 2, 3)

        x = F.pad(x, [0, self.padding_z, 0, self.padding_y, 0, self.padding_x]) # padding (0, self.padding) for z dim

        # x.shape: [16, 8, 96, 96, 84]

        for i in range(self.num_fourier_layers-1):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2 + x
            x = F.leaky_relu(x, negative_slope=self.ALPHA)
            # x = F.gelu(x)

        x1 = self.convs[-1](x)
        x2 = self.ws[-1](x)
        x = x1 + x2 + x

        _,_,n,m,l = x.shape
        x = x[..., :n-self.padding_x, :m-self.padding_y, :l-self.padding_z]
        
        # x.shape: [16, 8, 96, 96, 64]

        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x) # shape: [16, 96, 96, 64, 128]

        x = F.leaky_relu(x, negative_slope=self.ALPHA)
        # x = F.gelu(x)

        x = self.fc2(x) # shape: [16, 96, 96, 64, 6]
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
