#!/bin/bash
PYTHON='/users/sagar/miniconda3/envs/open_world_prototype/bin/python'
export CUDA_VISIBLE_DEVICES=0

hostname
nvidia-smi

# Get unique log file
SAVE_DIR=/work/sagar/open_set_recognition/dev_outputs/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -m methods.tests.openset_test_imagenet
#> ${SAVE_DIR}logfile_${EXP_NUM}.out