#!/bin/bash
out_file='train_ete.txt'

docker run -v /home/chenkaim/scripts/models/EM3d-surrogate/FNO3d:/workspace \
           -v /media/lts0/chenkaim/:/media/lts0/chenkaim/ \
           -v /media/ps2/chenkaim/:/media/ps2/chenkaim/ \
           --privileged --gpus all -tid --ipc=host --name EM3d_ete --rm chenkaim_torch:torch2.2 bash -c\
            "cd train_WaveYNet

            python3 train_ete_FNO_DDP.py \
                --model_name superpixel_DDP_ete_TiO2_SM_mod2_periodic_small \
                --model_file FNO3d_SM_ete \
                --model_saving_path /media/lts0/chenkaim/checkpoints/EM3d/ \
                --data_folder /media/lts0/chenkaim/3d_data/SR_half_periodic_TiO2 \
                --f_modes_x 24 \
                --f_modes_y 6 \
                --f_modes_z 8 \
                --periodic_x 0 \
                --periodic_y 1 \
                --periodic_z 0 \
                --HIDDEN_DIM 16 \
                --HIDDEN_DIM_freq 32 \
                --num_fourier_layers 4 \
                --domain_sizex 96 \
                --domain_sizey 16 \
                --domain_sizez 32 \
                --ALPHA 0.01 \
                --x_padding 20 \
                --y_padding 0 \
                --z_padding 10 \
                --start_lr 3e-4 \
                --end_lr 1e-5 \
                --weight_decay 0 \
                --continue_train 0 \
                --epoch 200 \
                --data_weight 1 \
                --inner_weight 1 \
                --phys_start_epoch 100 \
                --batch_size 32 \
                --world_size 4 \
                --gpus 3,4,5,6 \
                --seed 0
                > ${out_file} 2>&1"

                # world_size 1
                # --total_sample_number 200000 \
                # --model_name fourier_robin_64_Si_only_1_500k_data_DDP \

