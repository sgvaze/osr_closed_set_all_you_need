#!/bin/bash
PYTHON='/users/sagar/miniconda3/envs/open_set_recognition/bin/python'
export CUDA_VISIBLE_DEVICES=0

# Get unique log file
SAVE_DIR=/work/sagar/open_set_recognition/dev_outputs/

LOSS='Softmax'          # For TinyImageNet, ARPLoss and Softmax loss have the same
                        # RandAug and Label Smoothing hyper-parameters, but different learning rates

# Fixed hyper params for both ARPLoss and Softmax
AUG_M=9
AUG_N=1
LABEL_SMOOTHING=0.9

# LR different for ARPLoss and Softmax
if [ $LOSS = "Softmax" ]; then
   LR=0.01
elif [ $LOSS = "ARPLoss" ]; then
   LR=0.001
fi

# tinyimagenet
for SPLIT_IDX in 0 1 2 3 4; do

  EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
  EXP_NUM=$((${EXP_NUM}+1))
  echo $EXP_NUM

  ${PYTHON} -m methods.ARPL.osr  --lr=${LR} --model='classifier32' --transform='rand-augment' --rand_aug_m=${AUG_M} --rand_aug_n=${AUG_N}  \
  --dataset='tinyimagenet' --image_size=64 --loss=${LOSS} --scheduler='cosine_warm_restarts_warmup' --label_smoothing=${LABEL_SMOOTHING} \
  --split_train_val='False' --batch_size=128 --num_workers=16 --max-epoch=600 --seed=0 --gpus 0 --weight_decay=1e-4 --num_restarts=2 --feat_dim=128 --split_idx=${SPLIT_IDX}  \
   > ${SAVE_DIR}logfile_${EXP_NUM}.out

done