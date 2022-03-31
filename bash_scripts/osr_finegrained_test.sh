#!/bin/bash
PYTHON='/users/sagar/miniconda3/envs/open_world_prototype/bin/python'
export CUDA_VISIBLE_DEVICES=0

hostname
nvidia-smi

# Get unique log file
SAVE_DIR=/work/sagar/open_set_recognition/dev_outputs/

# SPECIFY PARAMS
DATASET='cub'
LOSS='Softmax'
EXP_ID='(29.07.2021_|_00.305)'

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -m methods.tests.openset_test_fine_grained --model='timm_resnet50_pretrained' --transform='rand-augment' \
             --loss=${LOSS} --exp_id=${EXP_ID} \
            --dataset=${DATASET} --image_size=448 \
            --batch_size=32 --num_workers=16 --gpus 0 --feat_dim=2048 \
#> ${SAVE_DIR}logfile_${EXP_NUM}.out