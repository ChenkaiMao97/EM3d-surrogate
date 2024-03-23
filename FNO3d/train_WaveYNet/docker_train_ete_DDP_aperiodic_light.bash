#!/bin/bash
out_file='train_ete_aperiodic.txt'

docker run -v /home/chenkaim/scripts/models/EM3d-surrogate/FNO3d:/workspace \
           -v /media/lts0/chenkaim/:/media/lts0/chenkaim/ \
           -v /media/ps2/chenkaim/:/media/ps2/chenkaim/ \
           --privileged --gpus all -tid --ipc=host --name EM3d_ete_light --rm chenkaim_torch:torch2.2 bash -c\
            "cd train_WaveYNet

            python3 train_ete_FNO_aperiodic_DDP.py \
                --model_name superpixel_ete_8freqs_conv_no_src_in_PML_10k \
                --model_file FNO3d_SM_ete_light \
                --model_saving_path /media/lts0/chenkaim/checkpoints/EM3d/ \
                --data_folder /media/lts0/chenkaim/3d_data/SR_aperiodic_TiO2_no_src_in_PML \
                --f_modes_x 20 \
                --f_modes_y 20 \
                --f_modes_z 10 \
                --periodic_x 0 \
                --periodic_y 0 \
                --periodic_z 0 \
                --HIDDEN_DIM 32 \
                --HIDDEN_DIM_freq 64 \
                --num_fourier_layers 6 \
                --domain_sizex 64 \
                --domain_sizey 64 \
                --domain_sizez 32 \
                --ALPHA 0.05 \
                --x_padding 10 \
                --y_padding 10 \
                --z_padding 10 \
                --start_lr 3e-5 \
                --end_lr 1e-6 \
                --weight_decay 0 \
                --continue_train 0 \
                --epoch 100 \
                --data_weight 1 \
                --inner_weight 1 \
                --phys_start_epoch 100 \
                --batch_size 16 \
                --world_size 1 \
                --gpus 4 \
                --seed 0
                > ${out_file} 2>&1"

                # world_size 1
                # --total_sample_number 200000 \
                # --model_name fourier_robin_64_Si_only_1_500k_data_DDP \

