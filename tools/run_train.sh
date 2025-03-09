#! /bin/bash

# nus mamba
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/lion_models/lion.yaml \
--extra_tag lion_bs2_lr75_1_14 \
--batch_size 2 --epochs 36 --max_ckpt_save_num 4 --workers 4 --sync_bn


# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
# --nproc_per_node=2 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
# --cfg_file ./cfgs/lion_models/fusion_lion.yaml \
# --extra_tag fusion_lion_bs2_lr75_1_14 \
# --batch_size 2 --epochs 36 --max_ckpt_save_num 4 --workers 1 --sync_bn


# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
# --nproc_per_node=2 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
# --cfg_file ./cfgs/lion_models/lion_hilbert.yaml \
# --extra_tag lion_hilbert_bs2_lr75_1_14 \
# --batch_size 2 --epochs 36 --max_ckpt_save_num 4 --workers 4 --sync_bn


# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
# --nproc_per_node=2 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
# --cfg_file ./cfgs/lion_models/lion_hilbert_es.yaml \
# --extra_tag lion_hilbert_es_bs2_lr75_1_14 \
# --batch_size 2 --epochs 36 --max_ckpt_save_num 4 --workers 4 --sync_bn --ckpt
