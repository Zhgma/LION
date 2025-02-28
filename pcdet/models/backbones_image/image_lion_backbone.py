from ..backbones_3d.lion_backbone_one_stride import LION3DBackboneOneStride

class LIONImageBackbone(LION3DBackboneOneStride):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__(model_cfg, input_channels, grid_size, **kwargs)
        