#!bin/bash

WORKSPACE="./"
DATA_DIR1="/root/autodl-nas/cnn/data/waveform.h5"
DATA_DIR2="/root/autodl-nas/cnn/data/waveform_se.h5"
DATA_DIR3="/root/autodl-tmp/demon.h5"
DATA_DIR4="/root/autodl-tmp/lofar.h5"
DATA_DIR5="/root/autodl-tmp/fusion.h5"
DATA_DIR6="/root/autodl-nas/cnn/data/waveform_all.h5"

python pytorch/main.py train \
    --workspace=$WORKSPACE \
    --hdf5_path=$DATA_DIR6 \
    --holdout_fold=4 \
    --model_type='Cnn10_gf' \
    --loss_type=clip_bce \
    --augmentation='none' \
    --learning_rate=1e-4 \
    --batch_size=16 \
    --stop_iteration=2000 \
    --feature_name='mel' \
    --learnable='none'
'''
for h in `seq 1 4`
do
python pytorch/main.py train \
    --workspace=$WORKSPACE \
    --hdf5_path=$DATA_DIR1 \
    --holdout_fold=1 \
    --model_type='Transfer_Cnn10_gf' \
    --pretrained_checkpoint_path='/root/autodl-nas/cnn/checkpoints/Cnn10_mAP=0.380.pth'\
    --loss_type=clip_bce \
    --augmentation='none' \
    --learning_rate=1e-4 \
    --batch_size=16 \
    --stop_iteration=2000 \
    --cuda \
    --wandb \
    --feature_name='mel' \
    --learnable='gabor+mel+stft'
done

for h in `seq 1 4`
do
    python pytorch/main.py train \
        --workspace=$WORKSPACE \
        --hdf5_path=$DATA_DIR1 \
        --holdout_fold=$h \
        --model_type='Cnn10_gf_no_specaug' \
        --loss_type=clip_bce \
        --augmentation='none' \
        --learning_rate=1e-4 \
        --batch_size=16 \
        --stop_iteration=2000 \
        --cuda \
        --wandb \
        --feature_name='mel' \
        --learnable='stft'
done

for h in `seq 1 4`
do
    python pytorch/main.py train \
        --workspace=$WORKSPACE \
        --hdf5_path=$DATA_DIR1 \
        --holdout_fold=$h \
        --model_type='Cnn10_fusion' \
        --loss_type=clip_bce \
        --augmentation='none' \
        --learning_rate=1e-4 \
        --batch_size=16 \
        --stop_iteration=2000 \
        --cuda \
        --wandb \
        --feature_name='mel' \
        --learnable='none'
done
'''