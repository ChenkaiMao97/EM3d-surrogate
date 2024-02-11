import  torch, os
from    torch import optim
import  numpy as np
import pandas as pd
import  argparse
from torch.utils.data import random_split, DataLoader

import sys 
sys.path.append("../models")

sys.path.append("../dataloaders")
from simulation_dataset_ete import SimulationDataset
sys.path.append("../utils")
from ete_physics import *
from plotting import plot_3slices

import timeit
from tqdm import tqdm
import gc

import matplotlib 
matplotlib.use('agg')
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from fvcore.nn import FlopCountAnalysis


argparser = argparse.ArgumentParser()

# general training args
argparser.add_argument('--epoch', type=int, help='epoch number', default=100)
argparser.add_argument('--batch_size', type=int, help='batch size', default=64)
argparser.add_argument("--data_folder", type=str, help='folder for the data', default="/scratch/groups/jonfan/UNet/data/data_generation_52_thick_8bar_Si/30k_new_wmin625")
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
argparser.add_argument("--HIDDEN_DIM", type=int, help='width of Unet, i.e. number of kernels of first block', default=64)
argparser.add_argument("--ALPHA", type=float, help="negative slope of leaky relu", default=0.05)
argparser.add_argument("--f_modes", type=int, help="number of lowest fourier terms to keep and transform", default=20)
argparser.add_argument("--num_fourier_layers", type=int, help="number of lowest fourier terms to keep and transform", default=10)
argparser.add_argument("--domain_sizex", type=int, help="number of pixels in x direction of subdomain", default=32)
argparser.add_argument("--domain_sizey", type=int, help="number of pixels in y direction of subdomain", default=32)
argparser.add_argument("--domain_sizez", type=int, help="number of pixels in z direction of subdomain", default=32)
argparser.add_argument("--z_padding", type=int, help="padding for z direction, assuming only z is aperiodic", default = 0)

# args for physics training:
argparser.add_argument("--phys_start_epoch", type=int, help="starting epoch of physical regularizer", default = 1)
argparser.add_argument("--ratio", type=float, help="relative weight of physical regularizer", default = 0.1)
argparser.add_argument("--inner_weight", type=float, help="weight for inner physics loss term", default = 0)
argparser.add_argument("--data_weight", type=float, help="weight for data loss term", default = 1)

# control parameters
argparser.add_argument("--continue_train", type=int, help = "if ==1, continue train from continue_epoch", default=0)
argparser.add_argument("--world_size", type=int, help="number of GPUs to use (will use ids 0, 1, ... world_size-1)", default=1)
argparser.add_argument("--gpus", type=str, help = "comma separated integers for available GPUs", default="0")
argparser.add_argument("--seed", type=int, help = "seed for repeated exps", default=42)


args = argparser.parse_args()  

if args.model_file == "FNO3d":
    from FNO3d import FNO_multimodal_3d
elif args.model_file == "FNO3d_SM":
    from FNO3d_SM import FNO_multimodal_3d
elif args.model_file == "FNO3d_inject_physics":
    from FNO3d_inject_physics import FNO_multimodal_3d
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

# def plot_helper(data,step,path):
#     plt.figure()
#     ax = plt.gca()
#     im = ax.imshow(data.detach().cpu().numpy())
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     plt.colorbar(im, cax=cax)
#     plt.title(f"epoch{step}")
#     plt.savefig(path, transparent=True)
#     plt.close()

def MAE_loss(a,b):
    return torch.mean(torch.abs(a-b))/torch.mean(torch.abs(b))

def train(rank, args, train_ds, test_ds, means):
    torch.cuda.set_device(rank)
    
    model_path = args.model_saving_path + args.model_name + \
                                          "_domain_size_" + str(args.domain_sizex) + "_"+ str(args.domain_sizey) + \
                                          "_fmodes_" + str(args.f_modes) + \
                                          "_flayers_" + str(args.num_fourier_layers) + \
                                          "_Hidden_" + str(args.HIDDEN_DIM) + \
                                          "_z_padding_" + str(args.z_padding) + \
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

    
    # use start_lr and end_lr to calculate lr update steps:
    total_steps = args.epoch*len(train_ds)/args.batch_size
    update_times = np.log(args.end_lr/args.start_lr)/np.log(0.99)
    lr_update_steps = int(total_steps/update_times)
    if rank==0:
        print(f"start_lr: {args.start_lr}, end_le: {args.end_lr}, total_steps: {total_steps}, lr_update_steps: {lr_update_steps}")

    start_epoch=0
    if (args.continue_train == 1):
        df = pd.read_csv(model_path + '/'+'df.csv')
        print("Restoring weights from ", model_path+"/best_model.pt", flush=True)
        checkpoint = torch.load(model_path+"/best_model.pt")
        start_epoch=checkpoint['epoch']
        print(f"start_epoch is {start_epoch}")
        model = checkpoint['model']
        optimizer = torch.optim.Adam(model.parameters(), lr=args.start_lr, weight_decay=args.weight_decay)
        optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'].state_dict())
        loss_fn = MAE_loss
        
    else:
        df = pd.DataFrame(columns=['epoch','train_data_loss', 'train_inner_loss', 'test_data_loss', 'test_inner_loss'])
        model = FNO_multimodal_3d(args)

        loss_fn = MAE_loss
        optimizer = torch.optim.Adam(model.parameters(), lr=args.start_lr, weight_decay=args.weight_decay)
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    if rank==0:
        tmp = filter(lambda x: x.requires_grad, model.parameters())
        num = sum(map(lambda x: np.prod(x.shape), tmp))
        print(model)
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
                find_unused_parameters=True)


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

    assert args.batch_size % args.world_size == 0
    each_GPU_batch_size = args.batch_size // args.world_size
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=each_GPU_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=each_GPU_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler=test_sampler)


    FLOPs_recorded = False

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

            y_batch_train, yeez_batch_train = sample_batched['field'].cuda(non_blocking=True), sample_batched['yeez'].cuda(non_blocking=True)

            # record FLOPs for once:
            if not FLOPs_recorded and rank==0:
                FLOPs_recorded = True
                flops = FlopCountAnalysis(model, (yeez_batch_train))
                print("flops per input device: ", flops.total()/1e9/args.batch_size, 'G')
                with open(model_path + '/'+'config.txt', 'a') as f:
                    f.write(f'\nFLOPs per input device: {flops.total()/1e9/args.batch_size}(G)')

            logits = model(yeez_batch_train)
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

            if (idx + 1) % 5 == 0 and rank==0:
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
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer,
                    'lr_scheduler': lr_scheduler
                 }
        torch.save(checkpoint, model_path+"/last_model.pt")


        # evaluation

        # to save time, use running average to approximate loss:
        train_data_loss = running_data_loss
        train_inner_loss = running_inner_loss

        test_data_loss = 0
        test_inner_loss = 0
        if step % 5 == 0:
            for idx, sample_batched in enumerate(test_loader):
                y_batch_test, yeez_batch_test = sample_batched['field'].cuda(non_blocking=True), sample_batched['yeez'].cuda(non_blocking=True)

                with torch.no_grad():
                    logits = model(yeez_batch_test)
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
                    test_inner_loss = torch.tensor(0.)

                    if rank==0 and idx == 0:
                        plot_3slices(logits[0,:,:,:,2].detach().cpu().numpy(), model_path+"/plots/epoch_"+str(step)+"_output_Ey_r.png", my_cm=plt.cm.seismic, cm_zero_center=False)
                        plot_3slices(y_batch_test[0,:,:,:,2].detach().cpu().numpy(), model_path+"/plots/epoch_"+str(step)+"_gt_Ey_r.png", my_cm=plt.cm.seismic)
                        plot_3slices(yeez_batch_test[0,:,:,:].detach().cpu().numpy(), model_path+"/plots/epoch_"+str(step)+"_yeez.png")
                        plot_3slices(y_batch_test[0,:,:,:,2].detach().cpu().numpy()-logits[0,:,:,:,2].detach().cpu().numpy(), model_path+"/plots/epoch_"+str(step)+"_error_Ey_r.png", my_cm=plt.cm.OrRd)
                        # plot_3slices(Ey_FD.real[0,:,:,:].detach().cpu().numpy(), model_path+"/plots/epoch_"+str(step)+"_Ey_FD.png", my_cm=plt.cm.seismic)
                        

            test_data_loss /= len(test_loader)
            test_inner_loss /= len(test_loader)
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
                                    'state_dict': model.state_dict(),
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

    ds = SimulationDataset(args.data_folder, total_sample_number = args.total_sample_number)
    means = ds.means
    train_ds, test_ds = random_split(ds, [int(0.9*len(ds)), len(ds) - int(0.9*len(ds))])

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8764'
    mp.spawn(train, nprocs=args.world_size, args=(args, train_ds, test_ds, means))

if __name__ == '__main__':
    torch.set_default_tensor_type(torch.FloatTensor)
    main(args)
