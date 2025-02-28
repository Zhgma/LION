from .transfusion import TransFusion
from .. import backbones_image
import torch
class TransFusionV2(TransFusion):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_topology = [
            'backbone_image',
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        ]
        self.module_list = self.build_networks()

    # def forward(self, batch_dict):
    #     past_allocated_memory = 0
    #     for cur_module in self.module_list:
    #         batch_dict = cur_module(batch_dict)
    #         allocated_memory = torch.cuda.memory_allocated(batch_dict['points'].device) / 1024 / 1024
    #         if cur_module.__class__.__name__ == 'FusionLION3DBackbone'or cur_module.__class__.__name__ == 'LION3DBackboneOneStride':
    #             print(f"{cur_module.__class__.__name__}:\t {allocated_memory-past_allocated_memory:.2f} MB")
    #         past_allocated_memory = allocated_memory
    #     max_memory_allocated = torch.cuda.max_memory_allocated(batch_dict['points'].device) / 1024 / 1024
    #     max_memory_reserved = torch.cuda.max_memory_reserved(batch_dict['points'].device) / 1024 / 1024
    #     allocated_memory = torch.cuda.memory_allocated(batch_dict['points'].device) / 1024 / 1024  # 已分配的显存
    #     reserved_memory = torch.cuda.memory_reserved(batch_dict['points'].device) / 1024 / 1024    # 已保留的显存
    #     print(f"Allocated memory:\t {allocated_memory:.2f} MB")
    #     print(f"Reserved memory:\t {reserved_memory:.2f} MB")
    #     print(f"Max allocated memory:\t {max_memory_allocated:.2f} MB")
    #     print(f"Max reserved memory:\t {max_memory_reserved:.2f} MB")
    #     print("\n**********************************************************************")
        

    #     if self.training:
    #         loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

    #         ret_dict = {
    #             'loss': loss
    #         }
    #         return ret_dict, tb_dict, disp_dict
    #     else:
    #         pred_dicts, recall_dicts = self.post_processing(batch_dict)
    #         return pred_dicts, recall_dicts

    def build_backbone_image(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_IMAGE', None) is None:
            return None, model_info_dict

        backbone_image_module = backbones_image.__all__[self.model_cfg.BACKBONE_IMAGE.NAME](
            model_cfg=self.model_cfg.BACKBONE_IMAGE,
            # input_channels=model_info_dict['num_bev_features']
        )
        model_info_dict['module_list'].append(backbone_image_module)
        # model_info_dict['num_bev_features'] = backbone_image_module.num_bev_features
        return backbone_image_module, model_info_dict
    