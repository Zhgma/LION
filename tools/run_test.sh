#! /bin/bash

# nus mamba
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=29988 test.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/lion_models/fusion_lion.yaml \
--extra_tag fusion_lion_bs_2_1_14 \
--ckpt ../output/cfgs/lion_models/fusion_lion/fusion_lion_bs_2_1_14/ckpt/checkpoint_epoch_36.pth \
--batch_size 2 --workers 1


# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
# --nproc_per_node=2 --master_port=29988 test.py  --tcp_port 29988  --launcher pytorch  \
# --cfg_file ./cfgs/lion_models/lion.yaml \
# --extra_tag lion_bs2_1_14 \
# --ckpt ../output/cfgs/lion_models/lion/lion_bs2_1_14/ckpt/checkpoint_epoch_36.pth \
# --batch_size 2 --workers 4


CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=29988 test.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/lion_models/lion_hilbert.yaml \
--extra_tag lion_hilbert_bs2_1_14 \
--ckpt ../output/cfgs/lion_models/lion_hilbert/lion_hilbert_bs2_1_14/ckpt/checkpoint_epoch_36.pth \
--batch_size 2 --workers 4


CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=29988 test.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/lion_models/lion_hilbert_es.yaml \
--extra_tag lion_hilbert_es_bs2_1_14 \
--ckpt ../output/cfgs/lion_models/lion_hilbert_es/lion_hilbert_es_bs2_1_14/ckpt/checkpoint_epoch_36.pth \
--batch_size 2 --workers 4
