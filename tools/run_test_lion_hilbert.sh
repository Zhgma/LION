#! /bin/bash

## nus mamba
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=29988 test.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/lion_models/lion_hilbert.yaml \
--extra_tag lion_hilbert_debug \
--ckpt ../output/cfgs/lion_models/lion_hilbert/lion_hilbert_bs2_1_14/ckpt/latest_model.pth \
--batch_size 2 --workers 4