#!/bin/bash

# This folder contains the pre-trained checkpoints. The contents
# of fsi-backup.tar.gz should be extracted here.
export results="/mnt/public/tmp/beckhamc/fewshot_infogan"
export viz_tmp="/mnt/home/viz/fewshot_augment/tmp"

# stylegan2-ada-pytorch is a required dependency
# The commit I have used is d4b2afe9c27e3c305b721bc886d2cb5229458eba
export STYLEGAN2_ADA_PYTORCH="/mnt/home/github/stylegan2-ada-pytorch"
# Add sg2-ada to pythonpath
export PYTHONPATH="${PYTHONPATH}:${STYLEGAN2_ADA_PYTORCH}"

# Experiments will be saved to this directory.
export SAVEDIR_BASE=$results

# Was originally used for multi-gpu support but all experiments
# in this paper suffice to run on a single GPU. Just leave this
# at 0 or 1 for now unless you know what you're doing.
export WORLD_SIZE=0

####################################
# Stage 1: pre-trained classifiers #
####################################

export EMNIST_PRETRAINED_CLS_S0="$results/tmp900_v2_emnist_classifier-mpqcau/b5938362192ebb80ad00ea3c8400ab09"
export EMNIST_PRETRAINED_CLS_S1="$results/tmp900_v2_emnist_classifier-equnss/2bff08e95901c703ca2e421ddc60a697"
export EMNIST_PRETRAINED_CLS_S2="$results/tmp900_v2_emnist_classifier-equnss/98501961e79364ea98dff672140bb6a9"
export EMNIST_PRETRAINED_CLS_S3="$results/tmp900_v2_emnist_classifier-equnss/e16987bb878bcdc0ebe4ad315cad93d8"
export EMNIST_PRETRAINED_CLS_S4="$results/tmp900_v2_emnist_classifier-equnss/8e97f21e968bd59f2e2f0a58eee59b89"

export OG_PRETRAINED_CLS_S0="$results/tmp900_v2_og_classifier-uhzmoh/d56103324cb7cb0c7f1919e3b8774122"
export OG_PRETRAINED_CLS_S1="$results/tmp900_v2_og_classifier-uhzmoh/a3ba245c62ea74d0d9fec8af7472f295"
export OG_PRETRAINED_CLS_S2="$results/tmp900_v2_og_classifier-uhzmoh/22de199d1bfdb8236c131d050fcbe0c1"
export OG_PRETRAINED_CLS_S3="$results/tmp900_v2_og_classifier-uhzmoh/616ae890e95dec4fe50d9e154e620f21"
export OG_PRETRAINED_CLS_S4="$results/tmp900_v2_og_classifier-uhzmoh/5ba0e0b9ff9fe3cf6fabec2281f61ee4"

export CIFAR100_PRETRAINED_CLS_S0="$results/train_cls_cifar100-cooqbo/718133e69bd46565429cb2d23d8d6d01"

#######################
# Stage 2a: train GAN #
#######################

export EMNIST_STAGE2A_S0="$results/tmp900_v2_emnist_gan_stage2-kntvml/40326686e8c299aef578468da172d240"
export EMNIST_STAGE2A_S1="$results/tmp900_v2_emnist_gan_stage2-syvlap/3fffb4a9481765856382faddd3f4e0ca"
export EMNIST_STAGE2A_S2="$results/tmp900_v2_emnist_gan_stage2-lythex/2a4e23666da3f6ae4e167f0134a8308d"
export EMNIST_STAGE2A_S3="$results/tmp900_v2_emnist_gan_stage2-kzbhvm/5c67d12e7726c62812adb90ec204ff46"
export EMNIST_STAGE2A_S4="$results/tmp900_v2_emnist_gan_stage2-xqqyxq/3a160bdddc934b8ae88e963fd87092c0"

export OG_STAGE2A_S0="$results/tmp1600_v2_omniglot_ngf1024_nofusedadam-nktore/54ee8fc5ebc23c37f5d50d07ce7001e0"
export OG_STAGE2A_S1="$results/tmp1600_v2_omniglot_ngf1024_nofusedadam-muvqcf/a90ea4c4dd681d42c4b6e46099c567a8"
export OG_STAGE2A_S2="$results/tmp1600_v2_omniglot_ngf1024_nofusedadam-oqnnpx/7d8fd0588405d4a72b23c2766a974cb2"
export OG_STAGE2A_S3="$results/tmp1600_v2_omniglot_ngf1024_nofusedadam-zepqic/4e23d328dbe669b052fab43c0a4bd0f2"
export OG_STAGE2A_S4="$results/tmp1600_v2_omniglot_ngf1024_nofusedadam-tvgmce/b1f9fdef26f08bea3cdb9a8e55e41d83"

##########################
# Stage 2b: finetune GAN #
##########################

export EMNIST_STAGE2B_K5_S0="$results/tmp1600_v2_emnist_ft_k5-mbtrag/df64004d0e963f114557b6a358e9c176"
export EMNIST_STAGE2B_K5_S1="$results/tmp1600_v2_emnist_ft_k5-hrzdmf/bc1dceb4c35e8c89be84b842da20296e"
export EMNIST_STAGE2B_K5_S2="$results/tmp1600_v2_emnist_ft_k5-gtrevv/88d209417afa1ab150356f723f5070d8"
export EMNIST_STAGE2B_K5_S3="$results/tmp1600_v2_emnist_ft_k5-kjnilc/5f1cbaf55d094ae4f0671d163d2e99a0"
export EMNIST_STAGE2B_K5_S4="$results/tmp1600_v2_emnist_ft_k5-luwhgw/6526b1853e384222fbea0ba14064bfad"

export EMNIST_STAGE2B_K10_S0="$results/tmp1600_v2_emnist_ft_k5-kjnewa/79c8327dd9d713b658785fcae5ff6bd3"
export EMNIST_STAGE2B_K10_S1="$results/tmp1600_v2_emnist_ft_k5-qbqrev/58f190f81995d792785d3fbd216a9b5d"
export EMNIST_STAGE2B_K10_S2="$results/tmp1600_v2_emnist_ft_k5-bhzdby/abb797ba2aed39e447b8b71810964d7c"
export EMNIST_STAGE2B_K10_S3="$results/tmp1600_v2_emnist_ft_k5-fhtdws/268dbd2fdc0bd598317bc8c3c658ad66"
export EMNIST_STAGE2B_K10_S4="$results/tmp1600_v2_emnist_ft_k5-dmwfce/0f8b6ab7669ee0544003ecac01e423a7"

export EMNIST_STAGE2B_K15_S0="$results/tmp1600_v2_emnist_ft_k15-swwgkv/44a9c311d5e71639f12bf21c9a5403e7"
export EMNIST_STAGE2B_K15_S1="$results/tmp1600_v2_emnist_ft_k15-mjejsz/c69ef06fc7aa60c319f86e866918b326"
export EMNIST_STAGE2B_K15_S2="$results/tmp1600_v2_emnist_ft_k15-sjeulz/c8ceec01abdb97a792ecab24282747d6"
export EMNIST_STAGE2B_K15_S3="$results/tmp1600_v2_emnist_ft_k15-aiicle/1c40a9696c723549e67a668fe86808bf"
export EMNIST_STAGE2B_K15_S4="$results/tmp1600_v2_emnist_ft_k15-xobkfp/3c846d7d997b82fc3dafbbc0bb31a2a5"

export EMNIST_STAGE2B_K25_S0="$results/tmp1600_v2_emnist_ft_k25-tctqrs/1c19414fe900f22798bb8ad69e967383"
export EMNIST_STAGE2B_K25_S1="$results/tmp1600_v2_emnist_ft_k25-msdvcx/b1c51fcf5e468a23d621c1e01c9ba15d"
export EMNIST_STAGE2B_K25_S2="$results/tmp1600_v2_emnist_ft_k25-cpixzr/1bcd59038ac07613debc126e590bc689"
export EMNIST_STAGE2B_K25_S3="$results/tmp1600_v2_emnist_ft_k25-apmcno/00dceb1e4b9b6e206f3c6608d1e984d1"
export EMNIST_STAGE2B_K25_S4="$results/tmp1600_v2_emnist_ft_k25-cqfzfm/96298f1f94b4c4a4884e7e03b4b1fb87"

export OG_STAGE2B_K5_S0="$results/tmp1600_v2_omniglot_ngf1024_ft_k5-orziev/8bcbf02cb6890d7025a3a33622a7fd26"
export OG_STAGE2B_K5_S1="$results/tmp1600_v2_omniglot_ngf1024_ft_k5-qecggw/36882ede3d319368e9f673f6305e2bb3"
export OG_STAGE2B_K5_S2="$results/tmp1600_v2_omniglot_ngf1024_ft_k5-tssyju/04ab4102c634462f5ccd9e25b8edee00"
export OG_STAGE2B_K5_S3="$results/tmp1600_v2_omniglot_ngf1024_ft_k5-mjqtne/4e7959b15f84258a1d75f954f80871b6"
export OG_STAGE2B_K5_S4="$results/tmp1600_v2_omniglot_ngf1024_ft_k5-uifvmw/d6ecd9636692562091643bb21816b433"

# Semi-supervised

export EMNIST_STAGE2B_K5_SEMI_S0="$results/tmp1600_v2_emnist_ft_k5_semi-dqdite/c0e0f7cae57930ca247b88102e96d9af"
export EMNIST_STAGE2B_K5_SEMI_S1="$results/tmp1600_v2_emnist_ft_k5_semi-fhbtcb/a79dd22f95c23f6a74f99391aac96006"
export EMNIST_STAGE2B_K5_SEMI_S2="$results/tmp1600_v2_emnist_ft_k5_semi-sryxgu/8ae3e3b6ed5ed14437f94693a055cd64"
export EMNIST_STAGE2B_K5_SEMI_S3="$results/tmp1600_v2_emnist_ft_k5_semi-emvexp/06d3dcb7cc521f6f530d76d509e4b828"
export EMNIST_STAGE2B_K5_SEMI_S4="$results/tmp1600_v2_emnist_ft_k5_semi-hnsdgc/dec2233ba95f59119966702c5b3a4063"

export EMNIST_STAGE2B_K5_SEMI_S0="$results/tmp1600_v2_emnist_ft_k10_semi-zvlhan/c0e0f7cae57930ca247b88102e96d9af"
export EMNIST_STAGE2B_K5_SEMI_S1="$results/tmp1600_v2_emnist_ft_k10_semi-iqidbj/a79dd22f95c23f6a74f99391aac96006"
export EMNIST_STAGE2B_K5_SEMI_S2="$results/tmp1600_v2_emnist_ft_k10_semi-nrkbge/8ae3e3b6ed5ed14437f94693a055cd64"
export EMNIST_STAGE2B_K5_SEMI_S3="$results/tmp1600_v2_emnist_ft_k10_semi-zbkunn/06d3dcb7cc521f6f530d76d509e4b828"
export EMNIST_STAGE2B_K5_SEMI_S4="$results/tmp1600_v2_emnist_ft_k10_semi-aywhxu/dec2233ba95f59119966702c5b3a4063"

# Semi-supervised with tuning of alpha

export EMNIST_STAGE2B_K5_SEMI_ALPHA_S0="$results/tmp1600_v2_emnist_ft_k5_semi_alpha-cfid-vpoacw/d2fa53df93a2ee38785e30278844bbcf"
export EMNIST_STAGE2B_K5_SEMI_ALPHA_S1="$results/tmp1600_v2_emnist_ft_k5_semi_alpha-cfid-fcqrna/19763e0566f6b80627f667ec4247c833"
export EMNIST_STAGE2B_K5_SEMI_ALPHA_S2="$results/tmp1600_v2_emnist_ft_k5_semi_alpha-cfid-uinizv/cb7899074d9e3aa0ec238bf588c06ab1"
export EMNIST_STAGE2B_K5_SEMI_ALPHA_S3="$results/tmp1600_v2_emnist_ft_k5_semi_alpha-cfid-tthrko/4de0eaeac61e0ba7a09e1d8f55527965"
export EMNIST_STAGE2B_K5_SEMI_ALPHA_S4="$results/tmp1600_v2_emnist_ft_k5_semi_alpha-cfid-phvbkc/efcfc61de17d3845e11ff7c0477fb90c"

export EMNIST_STAGE2B_K10_SEMI_ALPHA_S0="$results/tmp1600_v2_emnist_ft_k10_semi_alpha-cfid-eekjjp/ecdf801488e3bddb860e0090a2dc24a0"
export EMNIST_STAGE2B_K10_SEMI_ALPHA_S1="$results/tmp1600_v2_emnist_ft_k10_semi_alpha-cfid-ppoykn/10e9e1cb001e53396587025b665e2ebc"
export EMNIST_STAGE2B_K10_SEMI_ALPHA_S2="$results/tmp1600_v2_emnist_ft_k10_semi_alpha-cfid-kxbyte/c69fd9d3de779dbb39408f614a474e85"
export EMNIST_STAGE2B_K10_SEMI_ALPHA_S3="$results/tmp1600_v2_emnist_ft_k10_semi_alpha-cfid-kvghzd/c1ad9f4f3f2e7e7ed1aeb247ddf28160"
export EMNIST_STAGE2B_K10_SEMI_ALPHA_S4="$results/tmp1600_v2_emnist_ft_k10_semi_alpha-cfid-vszqdi/cd76e4d53a6bfd4732eb58393a44b85f"

export EMNIST_STAGE2B_K15_SEMI_ALPHA_S0="$results/tmp1600_v2_emnist_ft_k15_semi_alpha-cfid-pgxmwz/84a38ce25237a932c72e4c3ec5057802"
export EMNIST_STAGE2B_K15_SEMI_ALPHA_S1="$results/tmp1600_v2_emnist_ft_k15_semi_alpha-cfid-cuhven/8a0d28e969344d520724e5e129ea5557"
export EMNIST_STAGE2B_K15_SEMI_ALPHA_S2="$results/tmp1600_v2_emnist_ft_k15_semi_alpha-cfid-xacslj/bf3496f939e7769e1dafa24d0af9cdb7"
export EMNIST_STAGE2B_K15_SEMI_ALPHA_S3="$results/tmp1600_v2_emnist_ft_k15_semi_alpha-cfid-yppfuw/782195e59e0a61d73e4442a202969f94"
export EMNIST_STAGE2B_K15_SEMI_ALPHA_S4="$results/tmp1600_v2_emnist_ft_k15_semi_alpha-cfid-emulfh/f0095fd0dc0ac30ca576a031183cd333"

export EMNIST_STAGE2B_K25_SEMI_ALPHA_S0="$results/tmp1600_v2_emnist_ft_k25_semi-alpha-mjsmoy/525d336088d2d96340d0e07a3ea0c2a8"
export EMNIST_STAGE2B_K25_SEMI_ALPHA_S1="$results/tmp1600_v2_emnist_ft_k25_semi-alpha-rvazvw/c7b4787d3ece2a36194efd8a9a33a9e8"
export EMNIST_STAGE2B_K25_SEMI_ALPHA_S2="$results/tmp1600_v2_emnist_ft_k25_semi-alpha-flkxxf/2ef65cdafeb9d3206fafba11e1a7125e"
export EMNIST_STAGE2B_K25_SEMI_ALPHA_S3="$results/tmp1600_v2_emnist_ft_k25_semi-alpha-yrlujf/7fd06002f002bcb438eaae0cc9ffde1c"
export EMNIST_STAGE2B_K25_SEMI_ALPHA_S4="$results/tmp1600_v2_emnist_ft_k25_semi-alpha-lnrupt/a3b9342d681db980b5bdc8a4977b6f7b"

export OG_STAGE2B_K5_SEMI_ALPHA_S0="$results/tmp1600_v2_omniglot_ngf1024_ft_k5_semi_alpha-foraqi/08e0e585610d7ef249375d0d88aceabc"
export OG_STAGE2B_K5_SEMI_ALPHA_S1="$results/tmp1600_v2_omniglot_ngf1024_ft_k5_semi_alpha-fugdwq/0176fdcdcda51d0edbb962ddc59396a5"
export OG_STAGE2B_K5_SEMI_ALPHA_S2="$results/tmp1600_v2_omniglot_ngf1024_ft_k5_semi_alpha-hzsquj/903dc73bec4d68f3566778067a52235f"
export OG_STAGE2B_K5_SEMI_ALPHA_S3="$results/tmp1600_v2_omniglot_ngf1024_ft_k5_semi_alpha-ixmncc/6646b0bbd60ccdddafd319381de2df60"
