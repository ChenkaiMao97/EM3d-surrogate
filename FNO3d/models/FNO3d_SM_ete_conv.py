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

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, ALPHA, stride=1, use_shortcut=True):
        super(BasicBlock3D, self).__init__()
        self.ALPHA = ALPHA
        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.use_shortcut = use_shortcut
        if self.use_shortcut:
            self.shortcut = nn.Sequential(nn.Identity())
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv3d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=False),
                    # nn.BatchNorm2d(self.expansion*planes)
                )

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), negative_slope=self.ALPHA)
        out = self.conv2(out)
        if self.use_shortcut:
            out += self.shortcut(x)
            out = F.leaky_relu(out, negative_slope=self.ALPHA)
        return out

class BasicBlock3D_Periodic(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, ALPHA, stride=1, use_shortcut=True, periodic=(0,0,0), final_activation=True):
        # periodic: whether periodic in x, y and z
        super(BasicBlock3D_Periodic, self).__init__()
        self.ALPHA = ALPHA

        self.padding1 = nn.Sequential(
                        nn.CircularPad3d((0,0,0,0,1,1)) if periodic[0] else nn.ZeroPad3d((0,0,0,0,1,1)),
                        nn.CircularPad3d((0,0,1,1,0,0)) if periodic[1] else nn.ZeroPad3d((0,0,1,1,0,0)),
                        nn.CircularPad3d((1,1,0,0,0,0)) if periodic[2] else nn.ZeroPad3d((1,1,0,0,0,0))
                       ) # pay attention to order!

        self.padding2 = nn.Sequential(
                        nn.CircularPad3d((0,0,0,0,1,1)) if periodic[0] else nn.ZeroPad3d((0,0,0,0,1,1)),
                        nn.CircularPad3d((0,0,1,1,0,0)) if periodic[1] else nn.ZeroPad3d((0,0,1,1,0,0)),
                        nn.CircularPad3d((1,1,0,0,0,0)) if periodic[2] else nn.ZeroPad3d((1,1,0,0,0,0))
                       ) # maybe we can reuse padding1, just copied to be safe

        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=3, stride=1, padding=0, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=1, padding=0, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)

        self.use_shortcut = use_shortcut
        if self.use_shortcut:
            self.shortcut = nn.Sequential(nn.Identity())
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv3d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=False),
                    # nn.BatchNorm2d(self.expansion*planes)
                )

        self.final_activation = final_activation

    def forward(self, x):
        out = F.leaky_relu(self.conv1(self.padding1(x)), negative_slope=self.ALPHA)
        out = self.conv2(self.padding2(out))
        if self.use_shortcut:
            out += self.shortcut(x)
            if self.final_activation:
                out = F.leaky_relu(out, negative_slope=self.ALPHA)
        return out

################################################################
# 3d fourier layers
################################################################

class Modulated_SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, ALPHA, modes1, modes2, modes3, periodic=False):
        super(Modulated_SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3
        self.periodic = periodic
        self.ALPHA = ALPHA

        self.conv1 = nn.Conv3d(2*self.in_channels, 2*self.out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2 = nn.Conv3d(2*self.in_channels, 2*self.out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv3d(2*self.in_channels, 2*self.out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv4 = nn.Conv3d(2*self.in_channels, 2*self.out_channels, kernel_size=5, stride=1, padding=2, bias=False)

        self.conv5 = nn.Conv3d(2*self.in_channels, 2*self.out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv6 = nn.Conv3d(2*self.in_channels, 2*self.out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv7 = nn.Conv3d(2*self.in_channels, 2*self.out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv8 = nn.Conv3d(2*self.in_channels, 2*self.out_channels, kernel_size=5, stride=1, padding=2, bias=False)

        # self.scale = (1 / (in_channels * out_channels))
        # self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype=torch.float32))
        # self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype=torch.float32))
        # self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype=torch.float32))
        # self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype=torch.float32))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,bioxyz->boxyz", input, weights)

    # def forward(self, x, mod1, mod2, mod3, mod4):
    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2+1, dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = torch.view_as_complex(self.conv1(torch.cat((x_ft[:, :, :self.modes1, :self.modes2, :self.modes3].real, x_ft[:, :, :self.modes1, :self.modes2, :self.modes3].imag), dim=1)).reshape([batchsize, self.out_channels, 2, self.modes1, self.modes2, self.modes3]).permute((0,1,3,4,5,2)).contiguous())
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = torch.view_as_complex(self.conv2(torch.cat((x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3].real, x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3].imag), dim=1)).reshape([batchsize, self.out_channels, 2, self.modes1, self.modes2, self.modes3]).permute((0,1,3,4,5,2)).contiguous())
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = torch.view_as_complex(self.conv3(torch.cat((x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3].real, x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3].imag), dim=1)).reshape([batchsize, self.out_channels, 2, self.modes1, self.modes2, self.modes3]).permute((0,1,3,4,5,2)).contiguous())
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = torch.view_as_complex(self.conv4(torch.cat((x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3].real, x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3].imag), dim=1)).reshape([batchsize, self.out_channels, 2, self.modes1, self.modes2, self.modes3]).permute((0,1,3,4,5,2)).contiguous())

        out_ft[:, :, self.modes1:-self.modes1, :self.modes2, :self.modes3] = torch.view_as_complex(self.conv5(torch.cat((x_ft[:, :, self.modes1:-self.modes1, :self.modes2, :self.modes3].real, x_ft[:, :, self.modes1:-self.modes1, :self.modes2, :self.modes3].imag), dim=1)).reshape([batchsize, self.out_channels, 2, x.size(-3)-2*self.modes1, self.modes2, self.modes3]).permute((0,1,3,4,5,2)).contiguous())
        out_ft[:, :, :self.modes1, self.modes2:-self.modes2, :self.modes3] = torch.view_as_complex(self.conv6(torch.cat((x_ft[:, :, :self.modes1, self.modes2:-self.modes2, :self.modes3].real, x_ft[:, :, :self.modes1, self.modes2:-self.modes2, :self.modes3].imag), dim=1)).reshape([batchsize, self.out_channels, 2, self.modes1, x.size(-2)-2*self.modes2, self.modes3]).permute((0,1,3,4,5,2)).contiguous())
        out_ft[:, :, self.modes1:-self.modes1, -self.modes2:, :self.modes3] = torch.view_as_complex(self.conv7(torch.cat((x_ft[:, :, self.modes1:-self.modes1, -self.modes2:, :self.modes3].real, x_ft[:, :, self.modes1:-self.modes1, -self.modes2:, :self.modes3].imag), dim=1)).reshape([batchsize, self.out_channels, 2, x.size(-3)-2*self.modes1, self.modes2, self.modes3]).permute((0,1,3,4,5,2)).contiguous())
        out_ft[:, :, -self.modes1:, self.modes2:-self.modes2, :self.modes3] = torch.view_as_complex(self.conv8(torch.cat((x_ft[:, :, -self.modes1:, self.modes2:-self.modes2, :self.modes3].real, x_ft[:, :, -self.modes1:, self.modes2:-self.modes2, :self.modes3].imag), dim=1)).reshape([batchsize, self.out_channels, 2, self.modes1, x.size(-2)-2*self.modes2, self.modes3]).permute((0,1,3,4,5,2)).contiguous())
        

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

        self.periodic = (args.periodic_x, args.periodic_y, args.periodic_z)

        self.hidden_freq = args.HIDDEN_DIM_freq
        
        self.padding_x = args.x_padding 
        self.padding_y = args.y_padding 
        self.padding_z = args.z_padding 

        self.sizex = args.domain_sizex
        self.sizey = args.domain_sizey
        self.sizez = args.domain_sizez
        
        self.input_data_channels = 6
        self.fc0 = nn.Linear(self.input_data_channels, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.num_fourier_layers = args.num_fourier_layers
        self.ALPHA = args.ALPHA

        self.convs = []
        self.ws = []

        for i in range(self.num_fourier_layers):
            self.convs.append(Modulated_SpectralConv3d(self.width, self.width, self.ALPHA, self.modes1, self.modes2, self.modes3, periodic=self.periodic))
            # self.ws.append(nn.Conv3d(self.width, self.width, 1))
            self.ws.append(BasicBlock3D_Periodic(self.width, self.width, self.ALPHA, use_shortcut=False, periodic=self.periodic))
        self.convs = nn.ModuleList(self.convs)
        self.ws = nn.ModuleList(self.ws)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, args.outc)

        # the modulation branch of the network:
        # self.m_basic1 = BasicBlock3D_Periodic(self.input_data_channels, self.width, self.ALPHA, 1, periodic=self.periodic)
        # self.m_basic2 = BasicBlock3D_Periodic(self.width, self.width, self.ALPHA, 1, periodic=self.periodic)
        # self.m_basic3 = BasicBlock3D_Periodic(self.width, self.width, self.ALPHA, 1, periodic=self.periodic)
        # self.m_bc1 = nn.Linear(int(self.width*self.sizex*self.sizey*self.sizez/128), self.hidden_freq)
        # self.m_bc2_1 = nn.Linear(self.hidden_freq, self.modes1*self.modes2*self.modes3)
        # self.m_bc2_2 = nn.Linear(self.hidden_freq, self.modes1*self.modes2*self.modes3)
        # self.m_bc2_3 = nn.Linear(self.hidden_freq, self.modes1*self.modes2*self.modes3)
        # self.m_bc2_4 = nn.Linear(self.hidden_freq, self.modes1*self.modes2*self.modes3)

    def forward(self, yeex, yeey, yeez):
        batch_size = yeez.shape[0] # shape: [bs, sx, sy, sz]
        grid = self.get_grid(yeez.shape, yeez.device) # shape: [bs, sx, sy, sz, 3]

        input_data = torch.cat((yeex[:,:,:,:,None], yeey[:,:,:,:,None], yeez[:,:,:,:,None], grid), dim=-1) # shape: [bs, sx, sy, sz, 6]

        # mod_data = input_data.permute(0, 4, 1, 2, 3)

        # mod = F.avg_pool3d(self.m_basic1(mod_data),(2,1,2))
        # mod = F.avg_pool3d(self.m_basic2(mod),(2,1,2))
        # mod = F.avg_pool3d(self.m_basic3(mod),2)
        # mod = F.leaky_relu(self.m_bc1(mod.reshape((batch_size, -1))), negative_slope=self.ALPHA)

        # mod1 = self.m_bc2_1(mod).reshape((batch_size, 1,1, self.modes1,self.modes2,self.modes3))
        # mod2 = self.m_bc2_2(mod).reshape((batch_size, 1,1, self.modes1,self.modes2,self.modes3))
        # mod3 = self.m_bc2_3(mod).reshape((batch_size, 1,1, self.modes1,self.modes2,self.modes3))
        # mod4 = self.m_bc2_4(mod).reshape((batch_size, 1,1, self.modes1,self.modes2,self.modes3))

        x = self.fc0(input_data) # shape: [16, 96, 96, 64, 8]
        x = x.permute(0, 4, 1, 2, 3)

        x = F.pad(x, [0, self.padding_z, 0, self.padding_y, 0, self.padding_x]) # padding (0, self.padding) for z dim

        # x.shape: [16, 8, 96, 96, 84]

        for i in range(self.num_fourier_layers-1):
            # x1 = self.convs[i](x, mod1, mod2, mod3, mod4)
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2 + x
            x = F.leaky_relu(x, negative_slope=self.ALPHA)
            # x = F.gelu(x)

        # x1 = self.convs[-1](x, mod1, mod2, mod3, mod4)
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
