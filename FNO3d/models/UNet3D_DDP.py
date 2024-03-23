import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np
import matplotlib
import matplotlib.pyplot as plt


class UNet3D(nn.Module):
    '''
    This class defines the neural network components used in the WaveY-Net's UNet
    architecture, and subsequently specifies how data passes through the model.
    '''
    def __init__(self, args):
        super(UNet3D, self).__init__()
        '''
        Defining the components used in the neural network
        '''
        self.alpha = args.ALPHA
        config = []
        for block in range(args.NUM_DOWN_CONV):

            # kernel_size = 2 if block==args.NUM_DOWN_CONV-1 else 3
            kernel_size = 3
            out_channels = (2**block)*args.HIDDEN_DIM
            if(block == 0):
                config += [
                    # out_c, in_c, k_t, k_h, k_w, stride, padding, bool_residual_add,  also only conv, without bias
                    ('conv3d', [out_channels,args.inc,kernel_size,kernel_size,kernel_size,1,1,True])
                ]
            else:
                config += [
                    # out_c, in_c, k_t, k_h, k_w, stride, padding
                    ('conv3d', [out_channels,out_channels//2,kernel_size,kernel_size,kernel_size,1,1,True]),
                ]
            config += [('bn', [out_channels]),
                       # alpha; if true then executes relu in place
                       ('leakyrelu', [self.alpha, False])
                       ]
            config += [('conv3d', [out_channels,out_channels,kernel_size,kernel_size,kernel_size,1,1,False]),
                       ('bn', [out_channels]),
                       ('leakyrelu', [self.alpha, False])
                       ]
            config += [('conv3d', [out_channels,out_channels,kernel_size,kernel_size,kernel_size,1,1,False]),
                       ('bn', [out_channels])]
            config += [('residual', []),
                       ('leakyrelu', [self.alpha, False])
                       ]
            config += [('conv3d', [out_channels,out_channels,kernel_size,kernel_size,kernel_size,1,1,True]),
                       ('bn', [out_channels]),
                       ('leakyrelu', [self.alpha, False])
                       ]
            config += [('conv3d', [out_channels,out_channels,kernel_size,kernel_size,kernel_size,1,1,False]),
                       ('bn', [out_channels]),
                       ('leakyrelu', [self.alpha, False])
                       ]
            config += [('conv3d', [out_channels,out_channels,kernel_size,kernel_size,kernel_size,1,1,False]),
                       ('bn', [out_channels])
                       ]
            config += [('residual', [])]
            config += [('leakyrelu', [self.alpha, False])]
            if block < 2:
                config += [('max_pool3d', [(1, 1, 2), (1, 1, 2), 0])] # kernel_size, padding
            else:
                config += [('max_pool3d', [(2, 2, 2), (2, 2, 2), 0])]
        for block in range(args.NUM_DOWN_CONV-1):
            out_channels = (2**(args.NUM_DOWN_CONV-block-2))*args.HIDDEN_DIM
            in_channels = out_channels*3
            config += [('upsample', [2])]
            config += [('conv3d', [out_channels,in_channels,kernel_size,kernel_size,kernel_size,1,1,True]),
                       ('bn', [out_channels]),
                       ('leakyrelu', [self.alpha, False])
                       ]
            config += [('conv3d', [out_channels,out_channels,kernel_size,kernel_size,kernel_size,1,1,False]),
                       ('bn', [out_channels]),
                       ('leakyrelu', [self.alpha, False])
                       ]
            config += [('conv3d', [out_channels,out_channels,kernel_size,kernel_size,kernel_size,1,1,False]),
                       ('bn', [out_channels])]
            config += [('residual', []),
                       ('leakyrelu', [self.alpha, False])
                       ]
            config += [('conv3d', [out_channels,out_channels,kernel_size,kernel_size,kernel_size,1,1,True]),
                       ('bn', [out_channels]),
                       ('leakyrelu', [self.alpha, False])
                       ]
            config += [('conv3d', [out_channels,out_channels,kernel_size,kernel_size,kernel_size,1,1,False]),
                       ('bn', [out_channels]),
                       ('leakyrelu', [self.alpha, False])
                       ]
            config += [('conv3d', [out_channels,out_channels,kernel_size,kernel_size,kernel_size,1,1,False]),
                       ('bn', [out_channels])
                       ]
            config += [('residual', [])]
            config += [('leakyrelu', [self.alpha, False])]

        # all the conv3d before are without bias, and this conv_b is with bias
        config += [('conv3d_b', [args.outc,args.HIDDEN_DIM,kernel_size,kernel_size,kernel_size,1,1])]
        self.config = config

        # this dict contains all tensors needed to be optimized
        # self.vars = nn.ParameterList()
        # running_mean and running_var
        # self.vars_bn = nn.ParameterList()
        self.loss_fn = nn.L1Loss()
        self.optimizer = None # will be initialized in waveynet_trainer.py
        self.lr_scheduler = None # will be initialized in waveynet_trainer.py
        self.residual_terms = None # to store the residual connect for addition later

        var_idx=0
        var_bn_idx=0

        for i, (name, param) in enumerate(self.config):
            if name == 'conv3d':
                # [ch_out, ch_in, kernelsz, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:5]), requires_grad=True)
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                setattr(self, f'param_{var_idx}', w)
                setattr(self, f'param_num', var_idx+1)
                var_idx=var_idx+1
                # self.vars.append(w)
            elif name == 'conv3d_b':
                # [ch_out, ch_in, kernelsz, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:5]), requires_grad=True)
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                setattr(self, f'param_{var_idx}', w)
                setattr(self, f'param_num', var_idx+1)
                var_idx=var_idx+1
                # self.vars.append(w)
                # [ch_out]
                setattr(self, f'param_{var_idx}', nn.Parameter(torch.zeros(param[0]), requires_grad=True))
                setattr(self, f'param_num', var_idx+1)
                var_idx=var_idx+1
                # self.vars.append(nn.Parameter(torch.zeros(param[0]), requires_grad=True))

            elif name == 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]), requires_grad=True)
                setattr(self, f'param_{var_idx}', w)
                setattr(self, f'param_num', var_idx+1)
                var_idx=var_idx+1
                # self.vars.append(w)
                # [ch_out]
                setattr(self, f'param_{var_idx}', nn.Parameter(torch.zeros(param[0]), requires_grad=True))
                setattr(self, f'param_num', var_idx+1)
                var_idx=var_idx+1
                # self.vars.append(nn.Parameter(torch.zeros(param[0]), requires_grad=True))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                setattr(self, f'param_bn_{var_bn_idx}', running_mean)
                var_bn_idx=var_bn_idx+1
                setattr(self, f'param_bn_{var_bn_idx}', running_var)
                var_bn_idx=var_bn_idx+1
                # self.vars_bn.extend([running_mean, running_var])

            elif name in ['tanh', 'relu', 'upsample', 'avg_pool3d', 'max_pool3d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid', 'residual']:
                continue
            else:
                raise NotImplementedError

    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name == 'conv3d':
                tmp = 'conv3d:(ch_in:%d, ch_out:%d, k:%dx%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5], param[6],)
                info += tmp + '\n'

            elif name == 'conv3d_b':
                tmp = 'conv3d:(ch_in:%d, ch_out:%d, k:%dx%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5], param[6],)
                info += tmp + '\n'

            elif name == 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'

            elif name == 'max_pool3d':
                tmp = 'max_pool3d:(k:%s, stride:%s, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid',\
                          'use_logits', 'bn', 'residual']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                # print(name)
                raise NotImplementedError

        return info

    def forward(self, x, vars=None, bn_training=True):
        '''
        Defining how the data flows through the components initialized in the
        __init__ function, defining the model
        '''

        # if vars is None:
        #     vars = self.vars
        x = x[:,None,:,:,:]

        idx = 0
        bn_idx = 0
        first_upsample=True

        blocks = []
        for name, param in self.config:
            if name == 'conv3d':
                # periodic padding
                kernel_width = param[2]
                if kernel_width % 2:
                    x_pad = torch.cat([x[:,:,:,:,x.shape[4]-int((kernel_width-1)/2):x.shape[4]],x], dim=4)
                    x_pad = torch.cat([x_pad, x[:,:,:,:,0:int((kernel_width-1)/2)]], dim=4)
                    x_pad = torch.cat([x_pad[:,:,:,x_pad.shape[3]-int((kernel_width-1)/2):x_pad.shape[3],:],x_pad], dim=3)
                    x_pad = torch.cat([x_pad, x_pad[:,:,:,0:int((kernel_width-1)/2),:]], dim=3)
                    x_pad = torch.cat([torch.zeros_like(x_pad[:,:,0:int((kernel_width-1)/2),:,:]), x_pad], dim=2)
                    x_pad = torch.cat([x_pad, torch.zeros_like(x_pad[:,:,0:int((kernel_width-1)/2),:,:])], dim=2)
                else:
                    x_pad = torch.cat([x[:,:,:,:,x.shape[4]-int(kernel_width/2):x.shape[4]],x], dim=4)
                    x_pad = torch.cat([x_pad, x[:,:,:,:,0:int((kernel_width/2)-1)]], dim=4)
                    x_pad = torch.cat([x_pad[:,:,:,x_pad.shape[3]-int(kernel_width/2):x_pad.shape[3],:],x_pad], dim=3)
                    x_pad = torch.cat([x_pad, x_pad[:,:,:,0:int((kernel_width/2)-1),:]], dim=3)
                    x_pad = torch.cat([torch.zeros_like(x_pad[:,:,0:int(kernel_width/2),:,:]), x_pad], dim=2)
                    x_pad = torch.cat([x_pad, torch.zeros_like(x_pad[:,:,0:int(kernel_width/2)-1,:,:])], dim=2)

                # w = vars[idx]
                w = getattr(self, f'param_{idx}')
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv3d(x_pad, w, stride=param[5], padding=0)
                idx += 1
                if param[7]:
                    self.residual_terms = torch.clone(x)

            elif name == 'conv3d_b':
                # periodic padding
                kernel_width = param[2]
                if kernel_width % 2:
                    x_pad = torch.cat([x[:,:,:,:,x.shape[4]-int((kernel_width-1)/2):x.shape[4]],x], dim=4)
                    x_pad = torch.cat([x_pad, x[:,:,:,:,0:int((kernel_width-1)/2)]], dim=4)
                    x_pad = torch.cat([x_pad[:,:,:,x_pad.shape[3]-int((kernel_width-1)/2):x_pad.shape[3],:],x_pad], dim=3)
                    x_pad = torch.cat([x_pad, x_pad[:,:,:,0:int((kernel_width-1)/2),:]], dim=3)
                    x_pad = torch.cat([torch.zeros_like(x_pad[:,:,0:int((kernel_width-1)/2),:,:]), x_pad], dim=2)
                    x_pad = torch.cat([x_pad, torch.zeros_like(x_pad[:,:,0:int((kernel_width-1)/2),:,:])], dim=2)
                else:
                    x_pad = torch.cat([x[:,:,:,:,x.shape[4]-int(kernel_width/2):x.shape[4]],x], dim=4)
                    x_pad = torch.cat([x_pad, x[:,:,:,:,0:int((kernel_width/2)-1)]], dim=4)
                    x_pad = torch.cat([x_pad[:,:,:,x_pad.shape[3]-int(kernel_width/2):x_pad.shape[3],:],x_pad], dim=3)
                    x_pad = torch.cat([x_pad, x_pad[:,:,:,0:int((kernel_width/2)-1),:]], dim=3)
                    x_pad = torch.cat([torch.zeros_like(x_pad[:,:,0:int(kernel_width/2),:,:]), x_pad], dim=2)
                    x_pad = torch.cat([x_pad, torch.zeros_like(x_pad[:,:,0:int(kernel_width/2)-1,:,:])], dim=2)

                # w, b = vars[idx], vars[idx + 1]
                w = getattr(self, f'param_{idx}')
                b = getattr(self, f'param_{idx + 1}')
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv3d(x_pad, w, b, stride=param[5], padding=0)
                idx += 2

            elif name == 'bn':
                # w, b = vars[idx], vars[idx + 1]
                w = getattr(self, f'param_{idx}')
                b = getattr(self, f'param_{idx + 1}')
                # running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                running_mean = getattr(self, f'param_bn_{bn_idx}')
                running_var = getattr(self, f'param_bn_{bn_idx + 1}')
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name == 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])

            elif name == 'upsample':
                if first_upsample:
                    first_upsample = False
                    x= blocks.pop()
                shortcut= blocks.pop()
                x = F.interpolate(x, size=(shortcut.shape[2],shortcut.shape[3],shortcut.shape[4]), mode='nearest')
                x = torch.cat([shortcut,x], dim=1) # batch, channels, t, h, w

            elif name == 'residual':
                x = x + self.residual_terms

            elif name == 'max_pool3d':
                blocks.append(x)
                x = F.max_pool3d(x, param[0], stride=param[1], padding=param[2])

            else:
                # print(name)
                raise NotImplementedError

        # make sure variable is used properly
        # assert idx == len(vars)
        # assert bn_idx == len(self.vars_bn)

        return x


    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for i in range(self.param_num):
                    p = getattr(self, f'param_{i}')
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override parameters since initial parameters will return with a generator.
        """
        # return self.vars
        return [getattr(self, f'param_{i}') for i in range(self.param_num)]