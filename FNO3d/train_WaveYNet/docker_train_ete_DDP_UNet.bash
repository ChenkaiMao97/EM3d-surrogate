#!/bin/bash
out_file='train_ete_UNet.txt'

docker run -v /home/chenkaim/scripts/models/EM3d-surrogate/FNO3d:/workspace \
           -v /media/lts0/chenkaim/:/media/lts0/chenkaim/ \
           --privileged --gpus all -tid --ipc=host --name EM3d_ete --rm rclupoiu/surrogate:latest_torch2 bash -c\
            "cd train_WaveYNet
            pip install fvcore einops

            python3 train_ete_FNO_DDP.py \
                --model_name superpixel_DDP_ete_UNet3d \
                --model_file UNet3d_ete \
                --model_saving_path /media/lts0/chenkaim/checkpoints/EM3d/ \
                --data_folder /media/lts0/chenkaim/3d_data/SR_half_periodic_version3 \
                --HIDDEN_DIM 32 \
                --resnet_block_groups 1 \
                --attn_dim_head 32 \
                --attn_heads 4 \
                --full_attn 0 \
                --flash_attn 0 \
                --ALPHA 0.01 \
                --start_lr 3e-4 \
                --end_lr 1e-5 \
                --weight_decay 1e-4 \
                --continue_train 0 \
                --epoch 100 \
                --data_weight 1 \
                --inner_weight 1 \
                --phys_start_epoch 100 \
                --batch_size 64 \
                --world_size 2 \
                --gpus 2,3 \
                --seed 0
                > ${out_file} 2>&1"

                # world_size 1
                # --total_sample_number 200000 \
                # --model_name fourier_robin_64_Si_only_1_500k_data_DDP \

