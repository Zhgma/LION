#! /bin/bash

## nus mamba
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=29988 test.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/lion_models/lion_mamba_nusc_8x_1f_1x_one_stride_128dim.yaml \
--extra_tag lion \
--ckpt ../output/cfgs/lion_models/lion_mamba_nusc_8x_1f_1x_one_stride_128dim/lion/ckpt/latest_model.pth \
--batch_size 2 --workers 4