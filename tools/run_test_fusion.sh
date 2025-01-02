#! /bin/bash

## nus mamba
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=29988 test.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/lion_models/cross_modal.yaml \
--extra_tag cross_modal \
--ckpt ../output/cfgs/lion_models/cross_modal/cross_modal_128dim/ckpt/latest_model.pth \
--batch_size 2 --workers 4