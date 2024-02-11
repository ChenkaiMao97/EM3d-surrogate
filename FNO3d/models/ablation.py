import  torch, os
from    torch import optim
import  numpy as np
import pandas as pd
import  argparse
from torch.utils.data import random_split, DataLoader
import sys 
sys.path.append("../util")
from FNO2d import FNO_multimodal_2d
# from SM_FNO2d_conv import FNO_multimodal_2d

from simulation_dataset_JAX_Si_only import SimulationDataset
from torch.cuda.amp import GradScaler, autocast
from Adam import Adam
import timeit
from tqdm import tqdm
import gc

import matplotlib 
matplotlib.use('agg')
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP

from fvcore.nn import FlopCountAnalysis

C_0 = 299792458.13099605
EPSILON_0 = 8.85418782e-12
MU_0 = 1.25663706e-6
dL = 6.25e-9
wavelength = 1050e-9
omega = 2 * np.pi * C_0 / wavelength
n_air=1.
n_Si=3.567390909090909
n_sub=1.45

def Hz_to_Ex(Hz_R, Hz_I, dL, omega, yeex, EPSILON_0 = EPSILON_0):
    # print("sh1: ", Hz_R.shape, yeex.shape)
    # x = 1 / 2 * (eps_grid[:, :, 1:, :] + eps_grid[:, :, 0:-1, :]) # Material averaging
    Ex_R = -(Hz_I[:, 1:, 1:-1] - Hz_I[:, 0:-1, 1:-1])/dL/omega/EPSILON_0/yeex[:, :-1] # The returned Ex is corresponding to Ex_ceviche[:, 0:-1]
    Ex_I = (Hz_R[:, 1:, 1:-1] - Hz_R[:, 0:-1, 1:-1])/dL/omega/EPSILON_0/yeex[:, :-1]

    return torch.stack((Ex_R, Ex_I), axis = 1)

def Hz_to_Ey(Hz_R, Hz_I, dL, omega, yeey, EPSILON_0 = EPSILON_0):
    # print("sh2: ", Hz_R.shape, yeey.shape)
    # y = 1 / 2 * (eps_grid[:, :, 1:, :] + torch.roll(eps_grid[:, :, 1:, :], 1, dims = 3))
    Ey_R = (Hz_I[:,1:-1,1:] - Hz_I[:,1:-1,:-1])/dL/omega/EPSILON_0/yeey[:, :, :-1] # The returned Ey is corresponding to Ey_ceviche[0:-1, :]
    Ey_I = -(Hz_R[:,1:-1,1:] - Hz_R[:,1:-1,:-1])/dL/omega/EPSILON_0/yeey[:, :, :-1]
    return torch.stack((Ey_R, Ey_I), axis = 1)

def E_to_Hz(Ey_R, Ey_I, Ex_R, Ex_I, dL, omega, MU_0 = MU_0):
    # print("sh3:", Ey_R.shape, Ey_I.shape, Ex_R.shape, Ex_I.shape)
    Hz_R = ((Ey_I[:, :, 1:] - Ey_I[:, :, 0:-1]) - (Ex_I[:, 1:, :] - Ex_I[:, 0:-1, :]))/dL/omega/MU_0
    Hz_I = -((Ey_R[:, :, 1:] - Ey_R[:, :, 0:-1]) - (Ex_R[:, 1:, :] - Ex_R[:, 0:-1, :]))/dL/omega/MU_0
    return torch.stack((Hz_R, Hz_I), axis = 1) # -Hy[1:, :]

def H_to_H(Hz_R, Hz_I, dL, omega, yeex, yeey, EPSILON_0 = EPSILON_0, MU_0 = MU_0):
    FD_Ex = Hz_to_Ex(Hz_R, Hz_I, dL, omega, yeex, EPSILON_0)
    FD_Ey = Hz_to_Ey(Hz_R, Hz_I, dL, omega, yeey, EPSILON_0)
    FD_H = E_to_Hz(FD_Ey[:, 0], FD_Ey[:, 1], FD_Ex[:, 0], FD_Ex[:, 1], dL, omega, MU_0)
    return FD_H

def H_to_bc(Hz_R, Hz_I, dL, omega, yeex, yeey, EPSILON_0 = EPSILON_0):
    Hz = Hz_R + 1j*Hz_I
    # x = 1 / 2 * (eps_grid[:, :, 1:, :] + eps_grid[:, :, 0:-1, :]) # Material averaging
    top_bc = (Hz[:,0,1:-1]-Hz[:,1,1:-1])+1j*2*np.pi*torch.sqrt(yeex[:,1,:])/wavelength*dL*1/2*(Hz[:,0,1:-1]+Hz[:,1,1:-1])
    bottom_bc = (Hz[:,-1,1:-1]-Hz[:,-2,1:-1])+1j*2*np.pi*torch.sqrt(yeex[:,-1,:])/wavelength*dL*1/2*(Hz[:,-1,1:-1]+Hz[:,-2,1:-1])
    left_bc = (Hz[:,1:-1,0]-Hz[:,1:-1,1])+1j*2*np.pi*torch.sqrt(yeey[:,:,1])/wavelength*dL*1/2*(Hz[:,1:-1,0]+Hz[:,1:-1,1])
    right_bc = (Hz[:,1:-1,-1]-Hz[:,1:-1,-2])+1j*2*np.pi*torch.sqrt(yeey[:,:,-1])/wavelength*dL*1/2*(Hz[:,1:-1,-1]+Hz[:,1:-1,-2])
    return torch.stack((torch.real(top_bc), torch.imag(top_bc), torch.real(bottom_bc), torch.imag(bottom_bc),\
                        torch.real(left_bc), torch.imag(left_bc), torch.real(right_bc), torch.imag(right_bc)), axis = 1)

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

def plot_helper(data,step,path):
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(data.detach().cpu().numpy())
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.title(f"epoch{step}")
    plt.savefig(path, transparent=True)
    plt.close()

def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    if args.gpu_id != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    ds = SimulationDataset(args.data_folder, total_sample_number = args.total_sample_number, data_mult=args.data_mult)
    torch.manual_seed(args.seed)
    train_ds, test_ds = random_split(ds, [int(0.9*len(ds)), len(ds) - int(0.9*len(ds))])
    del ds
    gc.collect()

    
    model_path = args.model_saving_path + args.model_name + \
                                          "_domain_size_" + str(args.domain_sizex) + "_"+ str(args.domain_sizey) + \
                                          "_fmodes_" + str(args.f_modes) + \
                                          "_flayers_" + str(args.num_fourier_layers) + \
                                          "_Hidden_" + str(args.HIDDEN_DIM) + \
                                          "_f_padding_" + str(args.f_padding) + \
                                          "_batch_size_" + str(args.batch_size) + "_lr_" + str(args.start_lr)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
        os.mkdir(model_path+'/plots')

    start_epoch=0
    if (args.continue_train == "True"):
        df = pd.read_csv(model_path + '/'+'df.csv')
        print("Restoring weights from ", model_path+"/best_model.pt", flush=True)
        checkpoint = torch.load(model_path+"/best_model.pt")
        start_epoch=checkpoint['epoch']
        print(f"start_epoch is {start_epoch}")
        model = checkpoint['model']
        optimizer = Adam(model.parameters(), lr=args.start_lr, weight_decay=args.weight_decay)
        optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'].state_dict())
        loss_fn = model.loss_fn
        
    else:
        df = pd.DataFrame(columns=['epoch','train_loss', 'train_phys_reg', 'test_loss', 'test_phys_reg'])

        if args.arch == "UNet":
            model = UNet(args)
        elif args.arch == "Fourier":
            model = FNO_multimodal_2d(args)
        else:
            raise("architecture {args.arch} hasn't been added!!")

        loss_fn = model.loss_fn
        optimizer = Adam(model.parameters(), lr=args.start_lr, weight_decay=args.weight_decay)
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    
    # use start_lr and end_lr to calculate lr update steps:
    total_steps = args.epoch*len(train_ds)/args.batch_size
    update_times = np.log(args.end_lr/args.start_lr)/np.log(0.99)
    lr_update_steps = int(total_steps/update_times)
    print(f"start_lr: {args.start_lr}, end_le: {args.end_lr}, total_steps: {total_steps}, lr_update_steps: {lr_update_steps}")


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
    
    model.cuda()
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False)

    FLOPs_recorded = False

    total_step = len(train_loader)
    gradient_count = 0
    best_loss = 1e4

    last_epoch_data_loss = 1
    last_epoch_physical_loss = 1 

    running_data_loss = torch.tensor([1]).cuda()
    running_inner_loss = torch.tensor([1]).cuda()
    running_bc_loss = torch.tensor([1]).cuda()
    
    for step in range(start_epoch, args.epoch):
        epoch_start_time = timeit.default_timer()
        print("epoch: ", step, flush=True)
        reg_norm = regConstScheduler(step, args, last_epoch_data_loss, last_epoch_physical_loss)
        # training
        for idx, sample_batched in enumerate(train_loader):
            gradient_count += 1
            optimizer.zero_grad(set_to_none=True)

            y_batch_train, yeex_batch_train, yeey_batch_train, top_bc_train, bottom_bc_train, left_bc_train, right_bc_train = sample_batched['field'].cuda(), sample_batched['yeex'].cuda(), sample_batched['yeey'].cuda(), sample_batched['top_bc'].cuda(),sample_batched['bottom_bc'].cuda(),sample_batched['left_bc'].cuda(),sample_batched['right_bc'].cuda()

            # record FLOPs for once:
            if not FLOPs_recorded:
                FLOPs_recorded=True
                flops = FlopCountAnalysis(model, (yeex_batch_train, yeey_batch_train, top_bc_train, bottom_bc_train, left_bc_train, right_bc_train))
                print("flops per input device: ", flops.total()/1e9/args.batch_size, 'G')
                with open(model_path + '/'+'config.txt', 'a') as f:
                    f.write(f'\nFLOPs per input device: {flops.total()/1e9/args.batch_size}(G)')

            logits = model(yeex_batch_train, yeey_batch_train, top_bc_train, bottom_bc_train, left_bc_train, right_bc_train)
            logits = logits.reshape(y_batch_train.shape)
            loss = args.data_weight*loss_fn(logits.contiguous().view(args.batch_size,-1), y_batch_train.contiguous().view(args.batch_size,-1))

            running_data_loss = 0.99*running_data_loss + 0.01*loss
            # comppute physical loss:
            # physical loss for the inner pixels:
            # FD_Hy = H_to_H(fields[:, 0], fields[:, 1], dL, omega, pattern)
            reg_inner = torch.tensor([0])
            phys_reg_bc = torch.tensor([0])
            if args.inner_phys_loss_weight > 0 or args.boundary_phys_loss_weight>0:
                fields = logits
                means = [1, 1]
                FD_Hy = H_to_H(fields[:, :, :, 0]*means[0], fields[:, :, :, 1]*means[1], dL, omega, yeex_batch_train[:,:,1:-1], yeey_batch_train[:,1:-1,:])
                
                # print("sh: ", FD_Hy.shape, logits.shape)
                phys_regR = loss_fn(FD_Hy[:, 0]/means[0], logits[:, 1:-1, 1:-1, 0])
                phys_regI = loss_fn(FD_Hy[:, 1]/means[1], logits[:, 1:-1, 1:-1, 1])
                reg_inner = 0.5*(phys_regR + phys_regI)
                running_inner_loss = 0.99*running_inner_loss + 0.01*reg_inner
                # physical loss for the boundary:
                bc_gt = H_to_bc(y_batch_train[:,:,:,0], y_batch_train[:,:,:,1],dL, omega, yeex_batch_train[:,:,1:-1], yeey_batch_train[:,1:-1,:])
                bc_pred = H_to_bc(logits[:,:,:,0], logits[:,:,:,1], dL,omega, yeex_batch_train[:,:,1:-1], yeey_batch_train[:,1:-1,:])
                phys_reg_bc = loss_fn(bc_gt, bc_pred)
                running_bc_loss = 0.99*running_bc_loss + 0.01*phys_reg_bc

                # if idx%100==0:
                #     print(f"data_loss: {loss.item()}, reg_inner {reg_inner.item()}, reg_bc {phys_reg_bc.item()}")
                
                loss += reg_norm*(args.inner_phys_loss_weight*reg_inner+args.boundary_phys_loss_weight*phys_reg_bc)
                
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            optimizer.step()

            if (idx + 1) % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}], data loss: {:.4f}, inner loss: {:.4f}, bc loss: {:.4f}'.format(
                    step + 1,
                    args.epoch,
                    idx + 1,
                    total_step,
                    loss.item(),
                    reg_inner.item(),
                    phys_reg_bc.item()
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
        train_loss = running_data_loss
        train_phys_reg = running_inner_loss

        # train_loss = 0
        # train_phys_reg = 0
        # for sample_batched in train_loader:
        #     y_batch_train, yeex_batch_train, yeey_batch_train, top_bc_train, bottom_bc_train, left_bc_train, right_bc_train = sample_batched['field'].cuda(), sample_batched['yeex'].cuda(), sample_batched['yeey'].cuda(), sample_batched['top_bc'].cuda(),sample_batched['bottom_bc'].cuda(),sample_batched['left_bc'].cuda(),sample_batched['right_bc'].cuda()
            
        #     # with autocast():
        #     with torch.no_grad():
        #         y_batch_train, yeex_batch_train, yeey_batch_train, top_bc_train, bottom_bc_train, left_bc_train, right_bc_train = sample_batched['field'].cuda(), sample_batched['yeex'].cuda(), sample_batched['yeey'].cuda(), sample_batched['top_bc'].cuda(),sample_batched['bottom_bc'].cuda(),sample_batched['left_bc'].cuda(),sample_batched['right_bc'].cuda()

        #         mean_bc = (torch.mean(torch.abs(top_bc_train), dim=(1,2,3), keepdim=True)+torch.mean(torch.abs(bottom_bc_train), dim=(1,2,3), keepdim=True)+\
        #                    torch.mean(torch.abs(left_bc_train), dim=(1,2,3), keepdim=True)+torch.mean(torch.abs(right_bc_train), dim=(1,2,3), keepdim=True))/4


        #         logits = model(yeex_batch_train, yeey_batch_train, top_bc_train, bottom_bc_train, left_bc_train, right_bc_train).reshape(y_batch_train.shape)
        #         logits = logits

        #         loss = loss_fn(logits.contiguous().view(args.batch_size,-1), y_batch_train.contiguous().view(args.batch_size,-1))
        #         train_loss += loss
                
        #         fields = logits*mean_bc
        #         means = [1, 1]
        #         FD_Hy = H_to_H(fields[:, :, :, 0]*means[0], fields[:, :, :, 1]*means[1], dL, omega, yeex_batch_train[:,:,1:-1], yeey_batch_train[:,1:-1,:])
                
        #         # print("sh: ", FD_Hy.shape, logits.shape)
        #         phys_regR = loss_fn(FD_Hy[:, 0]/means[0], logits[:, 1:-1, 1:-1, 0])
        #         phys_regI = loss_fn(FD_Hy[:, 1]/means[1], logits[:, 1:-1, 1:-1, 1])

        #         train_phys_reg += 0.5*(phys_regR + phys_regI)
                
        # train_loss /= len(train_loader)
        # train_phys_reg /= len(train_loader)

        test_loss = 0
        test_phys_reg = 0
        for idx, sample_batched in enumerate(test_loader):
            y_batch_test, yeex_batch_test, yeey_batch_test, top_bc_test, bottom_bc_test, left_bc_test, right_bc_test = sample_batched['field'].cuda(), sample_batched['yeex'].cuda(), sample_batched['yeey'].cuda(), sample_batched['top_bc'].cuda(),sample_batched['bottom_bc'].cuda(),sample_batched['left_bc'].cuda(),sample_batched['right_bc'].cuda()

            with torch.no_grad():
                logits, preprocessed = model(yeex_batch_test, yeey_batch_test, top_bc_test, bottom_bc_test, left_bc_test, right_bc_test, output_init=True)
                logits = logits.reshape(y_batch_test.shape)

                loss = loss_fn(logits.contiguous().view(args.batch_size,-1), y_batch_test.contiguous().view(args.batch_size,-1))

                test_loss += loss
                # test_phys_reg += 0.5*(phys_regR + phys_regI)

                fields = logits
                means = [1, 1]
                FD_Hy = H_to_H(fields[:, :, :, 0]*means[0], fields[:, :, :, 1]*means[1], dL, omega, yeex_batch_test[:,:,1:-1], yeey_batch_test[:,1:-1,:])
                
                # print("sh: ", FD_Hy.shape, logits.shape)
                phys_regR = loss_fn(FD_Hy[:, 0]/means[0], logits[:, 1:-1, 1:-1, 0])
                phys_regI = loss_fn(FD_Hy[:, 1]/means[1], logits[:, 1:-1, 1:-1, 1])

                test_phys_reg += 0.5*(phys_regR + phys_regI)

                if idx == 0:
                    plot_helper(logits[0,:,:,0], step, model_path+"/plots/epoch_"+str(step)+"_output.png")
                    plot_helper(y_batch_test[0,:,:,0], step, model_path+"/plots/epoch_"+str(step)+"_gt.png")
                    plot_helper(yeex_batch_test[0,:,:], step, model_path+"/plots/epoch_"+str(step)+"_yeex.png")
                    plot_helper(preprocessed[0,:,:,0], step, model_path+"/plots/epoch_"+str(step)+"_init.png")
                    plot_helper(y_batch_test[0,:,:,0]-logits[0,:,:,0], step, model_path+"/plots/epoch_"+str(step)+"_error.png")
                    

        test_loss /= len(test_loader)
        test_phys_reg /= len(test_loader)
        last_epoch_data_loss = test_loss
        last_epoch_physical_loss = test_phys_reg.detach().clone()
        # test_phys_reg *= reg_norm
    
        print('train loss: %.5f, test loss: %.5f' % (train_loss, test_loss), flush=True)
        new_df = pd.DataFrame([[step+1,str(lr_scheduler.get_last_lr()),train_loss.item(), train_phys_reg.item(), test_loss.item(), test_phys_reg.item()]], \
                            columns=['epoch', 'lr', 'train_loss', 'train_phys_reg', 'test_loss', 'test_phys_reg'])
        df = pd.concat([df,new_df])

        df.to_csv(model_path + '/'+'df.csv',index=False)

        if(test_loss<best_loss):
            best_loss = test_loss
            checkpoint = {
                            'epoch': step,
                            'model': model,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer,
                            'lr_scheduler': lr_scheduler
                         }
            torch.save(checkpoint, model_path+"/best_model.pt")
        epoch_stop_time = timeit.default_timer()
        print("epoch run time:", epoch_stop_time-epoch_start_time)

if __name__ == '__main__':
    torch.set_default_tensor_type(torch.FloatTensor)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=100)
    # argparser.add_argument('--imgsz', type=tuple, help='imgsz', default=(64,256))
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--outc', type=int, help='outc', default=2)
    argparser.add_argument('--batch_size', type=int, help='batch size', default=64)
    argparser.add_argument('--start_lr', type=float, help='initial learning rate', default=3e-4)
    argparser.add_argument('--end_lr', type=float, help='final learning rate', default=1e-5)
    argparser.add_argument("--data_folder", type=str, help='folder for the data', default="/scratch/groups/jonfan/UNet/data/data_generation_52_thick_8bar_Si/30k_new_wmin625")
    argparser.add_argument("--data_mult", type=float, help="multiplier for the data", default = 1)
    argparser.add_argument("--total_sample_number", type=int, help="total number of training and testing samples to take from the npy file (in case you don't want to use all the data there)", default=None)
    argparser.add_argument("--arch", type=str, help='architecture of the learner', default="Fourier")
    # argparser.add_argument('--NUM_DOWN_CONV', type=int, help='number of down conv blocks in Unet', default=6)
    argparser.add_argument("--HIDDEN_DIM", type=int, help='width of Unet, i.e. number of kernels of first block', default=64)
    argparser.add_argument("--model_saving_path", type=str, help="the root dir to save checkpoints", default="") 
    argparser.add_argument("--model_name", type=str, help="name for the model, used for storing under the model_saving_path", default="test")
    argparser.add_argument("--continue_train", type=str, help = "if true, continue train from continue_epoch", default='False')
    argparser.add_argument("--ALPHA", type=float, help="negative slope of leaky relu", default=0.05)
    argparser.add_argument("--weight_decay", type=float, help="l2 regularization coeff", default=1e-4)
    argparser.add_argument("--ratio", type=float, help="relative weight of physical regularizer", default = 0.5)
    argparser.add_argument("--phys_start_epoch", type=int, help="starting epoch of physical regularizer", default = 1)
    argparser.add_argument("--kernel_size", type=int, help="conv layer kernel size", default=3)
    argparser.add_argument("--f_modes", type=int, help="number of lowest fourier terms to keep and transform", default=20)
    argparser.add_argument("--num_fourier_layers", type=int, help="number of lowest fourier terms to keep and transform", default=10)
    argparser.add_argument("--domain_sizex", type=int, help="number of pixels in x direction of subdomain", default=32)
    argparser.add_argument("--domain_sizey", type=int, help="number of pixels in y direction of subdomain", default=32)
    argparser.add_argument("--f_padding", type=int, help="padding for non-periodic b.c.", default = 0)

    argparser.add_argument("--inner_phys_loss_weight", type=float, help="weight for inner physics loss term", default = 0)
    argparser.add_argument("--boundary_phys_loss_weight", type=float, help="weight for bounadry physics loss term", default = 0)
    argparser.add_argument("--data_weight", type=float, help="weight for data loss term", default = 1)
    

    #args relating to distributed training and memory 
    argparser.add_argument("--world_size", type=int, help="number of GPUs to use (will use ids 0, 1, ... world_size-1)", default=1)
    argparser.add_argument("--gpu_id", type=int, help = "if specified, only use that gpu (only for single GPU training)", default=-1)
    
    argparser.add_argument("--seed", type=int, help = "seed for repeated exps", default=42)
    

    args = argparser.parse_args()  

    #os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    main(args)
