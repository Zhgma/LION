#! /bin/bash

## nus mamba
# CUDA_VISIBLE_DEVICES=0 python \
# train.py \
# --cfg_file ./cfgs/lion_models/cross_modal.yaml \
# --extra_tag cross_modal \
# --batch_size 1 --epochs 36 --max_ckpt_save_num 4 --workers 1 --sync_bn
# nus mamba
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/lion_models/fusion_lion.yaml \
--extra_tag fusion_lion_bs_2_1_14 \
--batch_size 2 --epochs 36 --max_ckpt_save_num 4 --workers 1 --sync_bn
# TORCH_DISTRIBUTED_DEBUG=DETAIL find_unused_parameters=True CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
# --nproc_per_node=1 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
# --cfg_file ./cfgs/lion_models/cross_modal.yaml \
# --extra_tag cross_modal \
# --batch_size 1 --epochs 36 --max_ckpt_save_num 4 --workers 1 --sync_bn


