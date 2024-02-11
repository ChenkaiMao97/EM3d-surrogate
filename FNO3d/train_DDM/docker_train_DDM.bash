#!/bin/bash
out_file='train_DDM.txt'

docker run -v /home/chenkaim/scripts/models/EM3d-surrogate/FNO3d:/workspace \
           -v /media/ps2/chenkaim/:/media/ps2/chenkaim/ \
           -v /media/ps1/chenkaim/:/media/ps1/chenkaim/ \
           --privileged --gpus all -tid --ipc=host --name EM3d_DDM --rm rclupoiu/surrogate:latest_torch2 bash -c\
            "cd train_DDM
            pip install fvcore h5py webdataset

            python3 train_DDM_FNO.py \
                --model_name EM3d_DDM_SM_Si_test \
                --model_file FNO3d_SM_DDM \
                --model_saving_path /media/ps2/chenkaim/checkpoints/EM3d \
                --data_folder /media/ps2/chenkaim/3d_data/periodic_grating_wl800_th600 \
                --f_modes 16 \
                --HIDDEN_DIM 16 \
                --num_fourier_layers 10 \
                --cube_size 64 \
                --ALPHA 0.1 \
                --padding 10 \
                --start_lr 3e-4 \
                --end_lr 3e-5 \
                --continue_train 0 \
                --epoch 500 \
                --batch_size 16 \
                --data_weight 1 \
                --inner_weight 1 \
                --phys_start_epoch 500 \
                --gpu_id 6 \
                --seed 0
                > ${out_file} 2>&1"

                # world_size 1
                # --total_sample_number 200000 \
                # --model_name fourier_robin_64_Si_only_1_500k_data_DDP \

