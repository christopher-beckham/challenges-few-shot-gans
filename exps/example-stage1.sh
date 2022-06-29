#!/bin/bash


python launch.py \
--tl=trainval_clf.py `#train launcher for classifier` \
--json_file=$EMNIST_PRETRAINED_CLS_S0/exp_dict.json `#pretrained classifier for emnist seed0` \
--savedir=/tmp/emnist-stage1-seed0 \
-d /mnt/public/datasets
