import  torch, os
from    torch import optim
import  numpy as np
import pandas as pd
import  argparse
from torch.utils.data import random_split, DataLoader

import sys 
sys.path.append("../models")

sys.path.append("../dataloaders")
from simulation_dataset_DDM import SimulationDataset
# from simulation_dataset_DDM_webds import SimulationDataset

sys.path.append("../utils")
from ete_physics import *
from plotting import plot_3slices

import timeit
from tqdm import tqdm
import gc
from functools import partial
from itertools import islice

import matplotlib 
matplotlib.use('agg')
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from fvcore.nn import FlopCountAnalysis
# import webdataset as wds

argparser = argparse.ArgumentParser()

# general training args
argparser.add_argument('--epoch', type=int, help='epoch number', default=100)
argparser.add_argument('--batch_size', type=int, help='batch size', default=64)
argparser.add_argument("--data_folder", type=str, help='folder for the data', default="/scratch/groups/jonfan/UNet/data/data_generation_52_thick_8bar_Si/30k_new_wmin625")
argparser.add_argument("--data_format", type=str, help='data format', default="npy")
argparser.add_argument("--bc_mult", type=float, help='multipler for bcs', default=1)

# argparser.add_argument("--data_path_train", type=str, help='folder for the data', default="")
# argparser.add_argument("--data_path_test", type=str, help='folder for the data', default="")
# argparser.add_argument("--ds_length", type=int, help='total samples in dataset', default=1000)


argparser.add_argument("--total_sample_number", type=int, help="total number of training and testing samples to take from the npy file (in case you don't want to use all the data there)", default=None)
argparser.add_argument("--model_saving_path", type=str, help="the root dir to save checkpoints", default="") 
argparser.add_argument("--model_file", type=str, help="the name of the model file to be imported", default="FNO3d") 
argparser.add_argument("--model_name", type=str, help="name for the model, used for storing under the model_saving_path", default="test")
argparser.add_argument('--outc', type=int, help='outc', default=6)

# args for optimizer:
argparser.add_argument('--start_lr', type=float, help='initial learning rate', default=3e-4)
argparser.add_argument('--end_lr', type=float, help='final learning rate', default=1e-5)
argparser.add_argument("--weight_decay", type=float, help="l2 regularization coeff", default=1e-4)

# args for FNO:
argparser.add_argument("--HIDDEN_DIM", type=int, help='hidden dimension for FFT modes', default=64)
argparser.add_argument("--HIDDEN_DIM_freq", type=int, help='hidden dimension for self modulaiton', default=64)
argparser.add_argument("--ALPHA", type=float, help="negative slope of leaky relu", default=0.05)
argparser.add_argument("--f_modes", type=int, help="number of lowest fourier terms to keep and transform", default=20)
argparser.add_argument("--num_fourier_layers", type=int, help="number of lowest fourier terms to keep and transform", default=10)
argparser.add_argument("--cube_size", type=int, help="side length of the cubic region", default=32)
argparser.add_argument("--padding", type=int, help="padding for all xyz directions", default = 0)

# args for physics training:
argparser.add_argument("--phys_start_epoch", type=int, help="starting epoch of physical regularizer", default = 1)
argparser.add_argument("--ratio", type=float, help="relative weight of physical regularizer", default = 0.1)
argparser.add_argument("--inner_weight", type=float, help="weight for inner physics loss term", default = 0)
argparser.add_argument("--data_weight", type=float, help="weight for data loss term", default = 1)
argparser.add_argument("--bc_weight", type=float, help="weight for boundary physics loss term", default = 0)

# control parameters
argparser.add_argument("--continue_train", type=int, help = "if ==1, continue train from continue_epoch", default=0)
argparser.add_argument("--world_size", type=int, help="number of GPUs to use (will use ids 0, 1, ... world_size-1)", default=1)
argparser.add_argument("--gpus", type=str, help = "comma separated integers for available GPUs", default="0")
argparser.add_argument("--seed", type=int, help = "seed for repeated exps", default=42)


args = argparser.parse_args()  

if args.model_file == "FNO3d_DDM":
    from FNO3d_DDM import FNO_multimodal_3d
elif args.model_file == "FNO3d_SM_DDM":
    from FNO3d_SM_DDM import FNO_multimodal_3d
elif args.model_file == "FNO3d_inject_physics_DDM":
    from FNO3d_inject_physics_DDM import FNO_multimodal_3d
else:
    raise ValueError(f"no model file named {args.model_file}")


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def regConstScheduler(epoch, args, last_epoch_data_loss, last_epoch_physical_loss):
    if(epoch<args.phys_start_epoch):
        return 0
    else:
        return args.ratio*last_epoch_data_loss/last_epoch_physical_loss

def plot_boundaries(bcs,path):
    fig, ax = plt.subplots(3,2)

    im = ax[0,0].imshow(bcs[:,:,0,0])
    divider = make_axes_locatable(ax[0,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    im = ax[0,1].imshow(bcs[:,:,1,1])
    divider = make_axes_locatable(ax[0,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    im = ax[1,0].imshow(bcs[:,:,2,2])
    divider = make_axes_locatable(ax[1,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    im = ax[1,1].imshow(bcs[:,:,3,3])
    divider = make_axes_locatable(ax[1,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    im = ax[2,0].imshow(bcs[:,:,4,4])
    divider = make_axes_locatable(ax[2,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    im = ax[2,1].imshow(bcs[:,:,5,5])
    divider = make_axes_locatable(ax[2,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.savefig(path, transparent=True, dpi=200)
    plt.close()

def MAE_loss(a,b):
    return torch.mean(torch.abs(a-b))/torch.mean(torch.abs(b))

def processing(sample, means, z_min=45, z_max=138, cube_size=64, wl=800e-9, dL=12.5e-9):
    eps, Ex, Ey, Ez = sample['epsilon_ddm.npy'], sample['0.0001ex.npy'], sample['0.0001ey.npy'], sample['0.0001ez.npy']
            
    yee = torch.from_numpy(eps).type(torch.float32)
    Ex = torch.from_numpy(Ex).type(torch.complex64)
    Ey = torch.from_numpy(Ey).type(torch.complex64)
    Ez = torch.from_numpy(Ez).type(torch.complex64)

    yeex, yeey, yeez = yee[0], yee[1], yee[2]
    # x, y starting point could be random
    rx, ry = np.random.randint(yeex.shape[0]), np.random.randint(yeex.shape[1])

    # z starting point is in range of (z_min, z_max-cube_size)
    rz = np.random.randint(z_min, z_max-cube_size)
    # apply a random roll:
    
    yeex = torch.roll(torch.roll(torch.roll(yeex, -rx, dims=0), -ry, dims=1), -rz, dims=2)[:cube_size,:cube_size,:cube_size]
    yeey = torch.roll(torch.roll(torch.roll(yeey, -rx, dims=0), -ry, dims=1), -rz, dims=2)[:cube_size,:cube_size,:cube_size]
    yeez = torch.roll(torch.roll(torch.roll(yeez, -rx, dims=0), -ry, dims=1), -rz, dims=2)[:cube_size,:cube_size,:cube_size]

    Ex = torch.roll(torch.roll(torch.roll(Ex, -rx, dims=0), -ry, dims=1), -rz, dims=2)[:cube_size,:cube_size,:cube_size]
    Ey = torch.roll(torch.roll(torch.roll(Ey, -rx, dims=0), -ry, dims=1), -rz, dims=2)[:cube_size,:cube_size,:cube_size]
    Ez = torch.roll(torch.roll(torch.roll(Ez, -rx, dims=0), -ry, dims=1), -rz, dims=2)[:cube_size,:cube_size,:cube_size]

    field = torch.stack((Ex.real/means[0], Ex.imag/means[1], Ey.real/means[2], Ey.imag/means[3], Ez.real/means[4], Ez.imag/means[5]), dim=-1)

    # top_bc0 =    1j*2*np.pi*np.sqrt(yeex[1:2, :])*dL/wl*1/2*(field_rot0[0:1, :]+field_rot0[1:2, :]) + field_rot0[0:1, :]-field_rot0[1:2, :]
    bc_x_p = torch.cat([torch.view_as_real((f[-1,: ,: ]-f[-2,: ,: ])+1/2*(f[-1,: ,: ]+f[-2,: ,: ])*1j*2*np.pi*np.sqrt(yeex[-1,: ,: ])*dL/wl) for f in (Ex, Ey, Ez)], dim=-1)/torch.tensor(means[None,None,:])
    bc_x_n = torch.cat([torch.view_as_real((f[0 ,: ,: ]-f[1 ,: ,: ])+1/2*(f[0 ,: ,: ]+f[1 ,: ,: ])*1j*2*np.pi*np.sqrt(yeex[1 ,: ,: ])*dL/wl) for f in (Ex, Ey, Ez)], dim=-1)/torch.tensor(means[None,None,:])
    bc_y_p = torch.cat([torch.view_as_real((f[: ,-1,: ]-f[: ,-2,: ])+1/2*(f[: ,-1,: ]+f[: ,-2,: ])*1j*2*np.pi*np.sqrt(yeey[: ,-1,: ])*dL/wl) for f in (Ex, Ey, Ez)], dim=-1)/torch.tensor(means[None,None,:])
    bc_y_n = torch.cat([torch.view_as_real((f[: ,0 ,: ]-f[: ,1 ,: ])+1/2*(f[: ,0 ,: ]+f[: ,1 ,: ])*1j*2*np.pi*np.sqrt(yeey[: ,1 ,: ])*dL/wl) for f in (Ex, Ey, Ez)], dim=-1)/torch.tensor(means[None,None,:])
    bc_z_p = torch.cat([torch.view_as_real((f[: ,: ,-1]-f[: ,: ,-2])+1/2*(f[: ,: ,-1]+f[: ,: ,-2])*1j*2*np.pi*np.sqrt(yeez[: ,: ,-1])*dL/wl) for f in (Ex, Ey, Ez)], dim=-1)/torch.tensor(means[None,None,:])
    bc_z_n = torch.cat([torch.view_as_real((f[: ,: ,0 ]-f[: ,: ,1 ])+1/2*(f[: ,: ,0 ]+f[: ,: ,1 ])*1j*2*np.pi*np.sqrt(yeez[: ,: ,1 ])*dL/wl) for f in (Ex, Ey, Ez)], dim=-1)/torch.tensor(means[None,None,:])

    bcs = torch.stack((bc_x_p, bc_x_n, bc_y_p, bc_y_n, bc_z_p, bc_z_n), dim = -2)

    return (field, bcs, yeex, yeey, yeez)

def my_split_by_node(urls, rank, world_size):
    # print("rank: ", rank, "world_size: ", world_size)
    yield from islice(urls, rank, None, world_size)

def train(rank, args, train_ds, test_ds, means):
    torch.cuda.set_device(rank)
    
    model_path = args.model_saving_path + '/' + args.model_name + \
                                                "_cube_size_" + str(args.cube_size) + \
                                                "_fmodes_" + str(args.f_modes) + \
                                                "_flayers_" + str(args.num_fourier_layers) + \
                                                "_Hidden_" + str(args.HIDDEN_DIM) + \
                                                "_padding_" + str(args.padding) + \
                                                "_batch_size_" + str(args.batch_size) + "_lr_" + str(args.start_lr)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
        os.mkdir(model_path+'/plots')

    dist.init_process_group(                                   
        backend='nccl',                                         
        init_method='env://',                                   
        world_size=args.world_size,                              
        rank=rank                                               
    )

    FLOPs_recorded = False
    
    # use start_lr and end_lr to calculate lr update steps:
    total_steps = args.epoch*len(train_ds)/args.batch_size
    # total_steps = args.epoch*args.ds_length*0.9/args.batch_size
    update_times = np.log(args.end_lr/args.start_lr)/np.log(0.99)
    lr_update_steps = int(total_steps/update_times)
    if rank==0:
        print(f"start_lr: {args.start_lr}, end_le: {args.end_lr}, total_steps: {total_steps}, lr_update_steps: {lr_update_steps}")

    start_epoch=0
    if (args.continue_train == 1):
        df = pd.read_csv(model_path + '/'+'df.csv')
        print("Restoring weights from ", model_path+"/best_model.pt", flush=True)
        checkpoint = torch.load(model_path+"/best_model.pt")
        start_epoch = checkpoint['epoch']+1
        print(f"start_epoch is {start_epoch}")
        model = checkpoint['model'].module
        optimizer = torch.optim.Adam(model.parameters(), lr=args.start_lr, weight_decay=args.weight_decay)
        optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'].state_dict())
        loss_fn = MAE_loss
        FLOPs_recorded = True
    else:
        df = pd.DataFrame(columns=['epoch','train_data_loss', 'train_inner_loss', 'test_data_loss', 'test_inner_loss'])
        model = FNO_multimodal_3d(args)

        loss_fn = MAE_loss
        optimizer = torch.optim.Adam(model.parameters(), lr=args.start_lr, weight_decay=args.weight_decay)
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    if rank==0:
        tmp = filter(lambda x: x.requires_grad, model.parameters())
        num = sum(map(lambda x: np.prod(x.shape), tmp))
        # print(model)
        print('Total trainable tensors:', num, flush=True)
        #for name, param in model.named_parameters():
        #    print(name, param.size())
        with open(model_path + '/'+'config.txt', 'w') as f:
            f.write('\n'.join(sys.argv[1:]))
            f.write(model.__str__())
            f.write(f'Total trainable tensors: {num}')
    
    model.cuda(rank)
    model.means = means

    model = DDP(model,
                device_ids=[rank],
                # find_unused_parameters=True,
                output_device=rank)

    each_GPU_batch_size = args.batch_size // args.world_size

    # train_sampler = wids.DistributedChunkedSampler(train_ds, chunksize=10, shuffle=True)
    # test_sampler = wids.DistributedChunkedSampler(test_ds, chunksize=10, shuffle=True)
    # processing_means = partial(processing, means=means)
    # my_partial_split_node = partial(my_split_by_node, rank=rank, world_size=args.world_size)
    # train_ds = wds.DataPipeline(
    #     wds.SimpleShardList(args.data_path_train),
    #     my_partial_split_node,
    #     wds.split_by_worker,
    #     wds.tarfile_to_samples(),
    #     wds.decode("torch"),
    #     wds.map(processing_means),
    #     wds.batched(int(args.batch_size/args.world_size))
    # )
    # test_ds = wds.DataPipeline(
    #     wds.SimpleShardList(args.data_path_test),
    #     my_partial_split_node,
    #     wds.split_by_worker,
    #     wds.tarfile_to_samples(),
    #     wds.decode("torch"),
    #     wds.map(processing_means),
    #     wds.batched(int(args.batch_size/args.world_size))
    # )
    # train_loader = wds.WebLoader(train_ds, num_workers=4)
    # test_loader = wds.WebLoader(test_ds, num_workers=4)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds,
        num_replicas=args.world_size,
        rank=rank,
        shuffle=True
    )

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_ds,
        num_replicas=args.world_size,
        rank=rank,
        shuffle=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=each_GPU_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=False,
        sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=each_GPU_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=False,
        sampler=test_sampler)

    # total_step = int(args.ds_length*0.9/args.batch_size)
    total_step = len(train_loader)
    gradient_count = 0
    best_loss = 1e4

    last_epoch_data_loss = 1.
    last_epoch_physical_loss = 1. 

    running_data_loss = 1.
    running_inner_loss = 10.
    
    for step in range(start_epoch, args.epoch):
        train_sampler.set_epoch(step)
        test_sampler.set_epoch(step)
        epoch_start_time = timeit.default_timer()
        if rank==0:
            print("epoch: ", step, flush=True)
        reg_norm = regConstScheduler(step, args, last_epoch_data_loss, last_epoch_physical_loss)
        # training

        for idx, sample_batched in enumerate(train_loader):
            gradient_count += 1
            optimizer.zero_grad(set_to_none=True)

            y_batch_train, bcs_batch_train, yeez_batch_train = sample_batched['field'].to(rank), sample_batched['bcs'].to(rank), sample_batched['yeez'].to(rank)
            
            # record FLOPs for once:
            if not FLOPs_recorded and rank==0:
                FLOPs_recorded = True
                flops = FlopCountAnalysis(model, (yeez_batch_train, bcs_batch_train))
                print("flops per input device: ", flops.total()/1e9/each_GPU_batch_size, 'G')
                with open(model_path + '/'+'config.txt', 'a') as f:
                    f.write(f'\nFLOPs per input device: {flops.total()/1e9/each_GPU_batch_size}(G)')

            logits = model(yeez_batch_train, bcs_batch_train)
            logits = logits.reshape(y_batch_train.shape) # [bs, sx, sy, sz, 6] (6 = x,y,z + 3 grids)
            data_loss = loss_fn(logits, y_batch_train)
            loss = args.data_weight*data_loss

            running_data_loss = 0.95*running_data_loss + 0.05*data_loss

            # comppute physical loss:
            # physical loss for the inner pixels:
            # Ex = logits[:, :, :, :, 0]*means[0] + 1j*logits[:, :, :, :, 1]*means[1]
            # Ey = logits[:, :, :, :, 2]*means[2] + 1j*logits[:, :, :, :, 3]*means[3]
            # Ez = logits[:, :, :, :, 4]*means[4] + 1j*logits[:, :, :, :, 5]*means[5]

            # Hx, Hy, Hz = E_to_H_batched_GPU(Ex, Ey, Ez, dL, omega, MU_0 = MU_0, bloch_vector=None)

            # eps_grid = (yeex_batch_train, yeey_batch_train, yeez_batch_train)

            # Ex_FD, Ey_FD, Ez_FD = H_to_E_batched_GPU(Hx, Hy, Hz, dL, omega, eps_grid, EPSILON_0 = EPSILON_0, bloch_vector=None)

            # # print("sh: ", FD_Hy.shape, logits.shape)
            # inner_loss = 1/6*(loss_fn(Ex_FD.real[:,:,:,1:-1]/means[0], logits[:, :, :, 1:-1, 0]) +\
            #                   loss_fn(Ex_FD.imag[:,:,:,1:-1]/means[1], logits[:, :, :, 1:-1, 1]) +\
            #                   loss_fn(Ey_FD.real[:,:,:,1:-1]/means[2], logits[:, :, :, 1:-1, 2]) +\
            #                   loss_fn(Ey_FD.imag[:,:,:,1:-1]/means[3], logits[:, :, :, 1:-1, 3]) +\
            #                   loss_fn(Ez_FD.real[:,:,:,1:-1]/means[4], logits[:, :, :, 1:-1, 4]) +\
            #                   loss_fn(Ez_FD.imag[:,:,:,1:-1]/means[5], logits[:, :, :, 1:-1, 5]))
            inner_loss=torch.tensor(0.)

            running_inner_loss = 0.95*running_inner_loss + 0.05*inner_loss

            # if idx%100==0:
            #     print(f"data_loss: {loss.item()}, reg_inner {reg_inner.item()}, reg_bc {phys_reg_bc.item()}")
            
            loss += reg_norm*args.inner_weight*inner_loss
                
            loss.backward()
            optimizer.step()

            if (idx + 1) % 2 == 0 and rank==0:
                print('Epoch [{}/{}], Step [{}/{}], data loss: {:.4f}, inner loss: {:.4f}'.format(
                    step + 1,
                    args.epoch,
                    idx + 1,
                    total_step,
                    loss.item(),
                    inner_loss.item(),
                    )
                )

            if gradient_count >= lr_update_steps:
                gradient_count = 0
                lr_scheduler.step()

        #Save the weights at the end of each epoch
        checkpoint = {
                    'epoch': step,
                    'model': model,
                    # 'state_dict': model.state_dict(),
                    'optimizer': optimizer,
                    'lr_scheduler': lr_scheduler
                 }
        torch.save(checkpoint, model_path+"/last_model.pt")


        # evaluation

        # to save time, use running average to approximate loss:
        train_data_loss = running_data_loss
        train_inner_loss = running_inner_loss

        test_data_loss = torch.tensor(0.).cuda()
        test_inner_loss = torch.tensor(0.).cuda()
        if step % 5 == 0:
            for idx, sample_batched in enumerate(test_loader):
                y_batch_test, bcs_batch_test, yeez_batch_test = sample_batched['field'].to(rank), sample_batched['bcs'].to(rank), sample_batched['yeez'].to(rank)
            
                with torch.no_grad():
                    logits = model(yeez_batch_test, bcs_batch_test)
                    logits = logits.reshape(y_batch_test.shape)

                    data_loss = loss_fn(logits, y_batch_test)

                    test_data_loss += data_loss

                    # Ex = logits[:, :, :, :, 0]*means[0] + 1j*logits[:, :, :, :, 1]*means[1]
                    # Ey = logits[:, :, :, :, 2]*means[2] + 1j*logits[:, :, :, :, 3]*means[3]
                    # Ez = logits[:, :, :, :, 4]*means[4] + 1j*logits[:, :, :, :, 5]*means[5]

                    # Hx, Hy, Hz = E_to_H_batched_GPU(Ex, Ey, Ez, dL, omega, MU_0 = MU_0, bloch_vector=None)

                    # eps_grid = (yeex_batch_test, yeey_batch_test, yeez_batch_test)

                    # Ex_FD, Ey_FD, Ez_FD = H_to_E_batched_GPU(Hx, Hy, Hz, dL, omega, eps_grid, EPSILON_0 = EPSILON_0, bloch_vector=None)

                    # # print("sh: ", FD_Hy.shape, logits.shape)
                    # test_inner_loss = 1/6*(loss_fn(Ex_FD.real[:,:,:,1:-1]/means[0], logits[:, :, :, 1:-1, 0]) +\
                    #                   loss_fn(Ex_FD.imag[:,:,:,1:-1]/means[1], logits[:, :, :, 1:-1, 1]) +\
                    #                   loss_fn(Ey_FD.real[:,:,:,1:-1]/means[2], logits[:, :, :, 1:-1, 2]) +\
                    #                   loss_fn(Ey_FD.imag[:,:,:,1:-1]/means[3], logits[:, :, :, 1:-1, 3]) +\
                    #                   loss_fn(Ez_FD.real[:,:,:,1:-1]/means[4], logits[:, :, :, 1:-1, 4]) +\
                    #                   loss_fn(Ez_FD.imag[:,:,:,1:-1]/means[5], logits[:, :, :, 1:-1, 5]))
                    test_inner_loss = torch.tensor(0.).cuda()

                    if rank==0 and idx == 0:
                        plot_3slices(logits[0,:,:,:,2].detach().cpu().numpy(), model_path+"/plots/epoch_"+str(step)+"_output_Ey_r.png", my_cm=plt.cm.seismic, cm_zero_center=False)
                        plot_3slices(y_batch_test[0,:,:,:,2].detach().cpu().numpy(), model_path+"/plots/epoch_"+str(step)+"_gt_Ey_r.png", my_cm=plt.cm.seismic)
                        plot_boundaries(bcs_batch_test[0].detach().cpu().numpy(), model_path+"/plots/epoch_"+str(step)+"_boundaries.png")
                        plot_3slices(yeez_batch_test[0,:,:,:].detach().cpu().numpy(), model_path+"/plots/epoch_"+str(step)+"_yeez.png")
                        plot_3slices(y_batch_test[0,:,:,:,2].detach().cpu().numpy()-logits[0,:,:,:,2].detach().cpu().numpy(), model_path+"/plots/epoch_"+str(step)+"_error_Ey_r.png", my_cm=plt.cm.OrRd)
                        # plot_3slices(Ey_FD.real[0,:,:,:].detach().cpu().numpy(), model_path+"/plots/epoch_"+str(step)+"_Ey_FD.png", my_cm=plt.cm.seismic)
                        

            test_data_loss /= idx+1
            test_inner_loss /= idx+1
            last_epoch_data_loss = test_data_loss
            last_epoch_physical_loss = test_inner_loss.detach().clone()
            # test_inner_loss *= reg_norm
        
            if rank==0:
                print('train loss: %.5f, test loss: %.5f' % (train_data_loss, test_data_loss), flush=True)
                new_df = pd.DataFrame([[step+1,str(lr_scheduler.get_last_lr()),train_data_loss.item(), train_inner_loss.item(), test_data_loss.item(), test_inner_loss.item()]], \
                                    columns=['epoch', 'lr', 'train_data_loss', 'train_inner_loss', 'test_data_loss', 'test_inner_loss'])
                df = pd.concat([df,new_df])

                df.to_csv(model_path + '/'+'df.csv',index=False)

                if(test_data_loss<best_loss):
                    best_loss = test_data_loss
                    checkpoint = {
                                    'epoch': step,
                                    'model': model,
                                    # 'state_dict': model.state_dict(),
                                    'optimizer': optimizer,
                                    'lr_scheduler': lr_scheduler
                                 }
                    torch.save(checkpoint, model_path+"/best_model.pt")
        if rank==0:
            epoch_stop_time = timeit.default_timer()
            print("epoch run time:", epoch_stop_time-epoch_start_time)

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    print(args)

    print("GPUs to use: ", args.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)

    ds = SimulationDataset(args.data_folder, total_sample_number = args.total_sample_number, cube_size=args.cube_size, bc_mult = args.bc_mult, data_format=args.data_format)
    means = ds.means
    train_ds, test_ds = random_split(ds, [int(0.9*len(ds)), len(ds) - int(0.9*len(ds))])

    assert args.batch_size % args.world_size == 0
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    mp.spawn(train, nprocs=args.world_size, args=(args, train_ds, test_ds, means))

if __name__ == '__main__':
    torch.set_default_tensor_type(torch.FloatTensor)
    main(args)
