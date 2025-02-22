#! /bin/bash

## nus mamba
# CUDA_VISIBLE_DEVICES=0 python \
# train.py \
# --cfg_file ./cfgs/lion_models/lion_hilbert.yaml \
# --extra_tag lion_hilbert \
# --batch_size 2 --epochs 36 --max_ckpt_save_num 4 --workers 4 --sync_bn
## nus mamba
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/lion_models/lion_hilbert.yaml \
--extra_tag lion_hilbert_bs4_1_14 \
--batch_size 2 --epochs 36 --max_ckpt_save_num 4 --workers 4 --sync_bn
