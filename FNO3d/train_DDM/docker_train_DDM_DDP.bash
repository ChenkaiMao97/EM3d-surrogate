#!/bin/bash
out_file='train_DDM_DDP.txt'

docker run -v /home/chenkaim/scripts/models/EM3d-surrogate/FNO3d:/workspace \
           -v /media/ps2/chenkaim/:/media/ps2/chenkaim/ \
           -v /media/lts0/chenkaim/:/media/lts0/chenkaim/ \
           --privileged --gpus all -tid --ipc=host --name EM3d_DDM --rm chenkaim_torch:torch2.2 bash -c\
            "cd /workspace/train_DDM
            pip install fvcore

            python3 train_DDM_FNO_DDP.py \
                --model_name EM3d_DDP_DDM_SM_TiO2 \
                --model_file FNO3d_SM_DDM \
                --model_saving_path /media/lts0/chenkaim/checkpoints/EM3d \
                --data_folder /media/ps2/chenkaim/3d_data/periodic_grating_wl800_th600_TiO2 \
                --data_format npy \
                --bc_mult 50 \
                --f_modes 10 \
                --HIDDEN_DIM 32 \
                --HIDDEN_DIM_freq 128 \
                --num_fourier_layers 10 \
                --cube_size 64 \
                --ALPHA 0.1 \
                --padding 10 \
                --start_lr 3e-4 \
                --end_lr 1e-5 \
                --continue_train 0 \
                --epoch 400 \
                --batch_size 32 \
                --data_weight 1 \
                --inner_weight 1 \
                --phys_start_epoch 400 \
                --world_size 4 \
                --gpus 3,4,5,6 \
                --seed 0
                > ${out_file} 2>&1"

                

            # python3 train_DDM_FNO_DDP.py \
            #     --model_name EM3d_DDP_DDM_SM_TiO2_test \
            #     --model_file FNO3d_SM_DDM \
            #     --model_saving_path /media/ps2/chenkaim/checkpoints/EM3d \
            #     --data_path_train '/media/ps2/chenkaim/3d_data_tar/periodic_grating_wl800_th600_TiO2/train/data_{0000..0022}.tar' \
            #     --data_path_test '/media/ps2/chenkaim/3d_data_tar/periodic_grating_wl800_th600_TiO2/test/data_{0000..0002}.tar' \
            #     --ds_length 5000 \
            #     --f_modes 16 \
            #     --HIDDEN_DIM 16 \
            #     --HIDDEN_DIM_freq 64 \
            #     --num_fourier_layers 10 \
            #     --cube_size 64 \
            #     --ALPHA 0.1 \
            #     --padding 10 \
            #     --start_lr 3e-4 \
            #     --end_lr 1e-5 \
            #     --continue_train 0 \
            #     --epoch 500 \
            #     --batch_size 32 \
            #     --data_weight 1 \
            #     --inner_weight 1 \
            #     --phys_start_epoch 500 \
            #     --world_size 4 \
            #     --gpus 3,4,5,6 \
            #     --seed 0
