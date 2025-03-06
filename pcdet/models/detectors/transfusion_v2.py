from .transfusion import TransFusion
from .. import backbones_image

class TransFusionV2(TransFusion):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_topology = [
            'backbone_image', 'neck',
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        ]
        self.module_list = self.build_networks()

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
    
    def build_neck(self,model_info_dict):
        if self.model_cfg.get('NECK', None) is None:
            return None, model_info_dict
        neck_module = backbones_image.img_neck.__all__[self.model_cfg.NECK.NAME](
            model_cfg=self.model_cfg.NECK
        )
        model_info_dict['module_list'].append(neck_module)

        return neck_module, model_info_dict