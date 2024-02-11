#!/bin/bash
out_file='train_ete.txt'

docker run -v /home/chenkaim/scripts/models/EM3d-surrogate/FNO3d:/workspace -v /media/ps2/chenkaim/:/media/ps2/chenkaim/ --privileged --gpus all -tid --ipc=host --name EM3d_ete --rm rclupoiu/surrogate:latest_torch2 bash -c\
            "cd train_WaveYNet
            pip install fvcore

            python3 train_ete_FNO.py \
                --model_name EM3d_ete_SM_TiO2 \
                --model_file FNO3d_inject_physics \
                --model_saving_path /media/ps2/chenkaim/checkpoints/EM3d \
                --data_folder /media/ps2/chenkaim/3d_data/periodic_grating_wl800_th600_TiO2 \
                --f_modes 12 \
                --HIDDEN_DIM 12 \
                --num_fourier_layers 4 \
                --domain_sizex 96 \
                --domain_sizey 96 \
                --domain_sizez 64 \
                --ALPHA 0.1 \
                --z_padding 10 \
                --start_lr 3e-4 \
                --end_lr 1e-5 \
                --continue_train 0 \
                --epoch 200 \
                --batch_size 8 \
                --data_weight 1 \
                --inner_weight 1 \
                --phys_start_epoch 50 \
                --gpu_id 7 \
                --seed 0
                > ${out_file} 2>&1"

                # world_size 1
                # --total_sample_number 200000 \
                # --model_name fourier_robin_64_Si_only_1_500k_data_DDP \

