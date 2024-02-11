#!/bin/bash
out_file='train_ete.txt'

docker run -v /home/chenkaim/scripts/models/EM3d-surrogate/FNO3d:/workspace -v /media/ps2/chenkaim/:/media/ps2/chenkaim/ --privileged --gpus all -tid --ipc=host --name EM3d_ete --rm rclupoiu/surrogate:latest_torch2 bash -c\
            "cd train_WaveYNet
            pip install fvcore

            python3 train_ete_FNO_DDP.py \
                --model_name EM3d_DDP_ete_Si_test \
                --model_file FNO3d_SM \
                --model_saving_path /media/ps2/chenkaim/checkpoints/EM3d \
                --data_folder /media/ps2/chenkaim/3d_data/periodic_grating_wl800_th600 \
                --f_modes 16 \
                --HIDDEN_DIM 16 \
                --num_fourier_layers 4 \
                --domain_sizex 96 \
                --domain_sizey 96 \
                --domain_sizez 64 \
                --ALPHA 0.01 \
                --z_padding 10 \
                --start_lr 3e-4 \
                --end_lr 1e-5 \
                --continue_train 0 \
                --epoch 200 \
                --data_weight 1 \
                --inner_weight 1 \
                --phys_start_epoch 50 \
                --batch_size 9 \
                --world_size 3 \
                --gpus 5,6,7 \
                --seed 0
                > ${out_file} 2>&1"

                # world_size 1
                # --total_sample_number 200000 \
                # --model_name fourier_robin_64_Si_only_1_500k_data_DDP \

