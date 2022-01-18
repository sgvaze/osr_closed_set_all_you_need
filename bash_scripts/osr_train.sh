#!/bin/bash
PYTHON='/users/sagar/miniconda3/envs/open_set_recognition/bin/python'
export CUDA_VISIBLE_DEVICES=0

# Get unique log file
SAVE_DIR=/work/sagar/open_set_recognition/dev_outputs/

LOSS='Softmax'
dataset='cifar-10-10'

for SPLIT_IDX in 0 1 2 3 4; do

  EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
  EXP_NUM=$((${EXP_NUM}+1))
  echo $EXP_NUM

  ${PYTHON} -m methods.ARPL.osr --dataset=$DATASET --loss=${LOSS} --use_default_parameters='True' \
  --num_workers=16 --gpus 0 --split_idx=${SPLIT_IDX} > ${SAVE_DIR}logfile_${EXP_NUM}.out

done