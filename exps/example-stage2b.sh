#!/bin/bash

python launch.py \
--json_file=$EMNIST_STAGE2B_K5_S1/exp_dict.json        `#read exp dict from original experiment` \
--tl=trainval.py                                    `#trainval to launch GAN pre-training experiments` \
--savedir=/tmp/emnist-stage2b-k5-seed1                 `#save experiment here` \
-d /mnt/public/datasets                             `#datasets will be downloaded here`
