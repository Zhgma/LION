from .mean_vfe import MeanVFE
from .pillar_vfe import PillarVFE, PillarVFE3D
from .dynamic_mean_vfe import DynamicMeanVFE
from .dynamic_pillar_vfe import DynamicPillarVFE, DynamicPillarVFE_3d,DynamicPillarVFESimple2D
from .image_vfe import ImageVFE
from .vfe_template import VFETemplate
from .dynamic_voxel_vfe import DynamicVoxelVFE
from .dynamic_voxel_vfe_v2 import DynamicVoxelVFE_V2

__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'PillarVFE': PillarVFE,
    'PillarVFE3D': PillarVFE3D,
    'ImageVFE': ImageVFE,
    'DynMeanVFE': DynamicMeanVFE,
    'DynPillarVFE': DynamicPillarVFE,
    'DynPillarVFE3D': DynamicPillarVFE_3d,
    'DynamicPillarVFESimple2D': DynamicPillarVFESimple2D,
    'DynamicVoxelVFE': DynamicVoxelVFE,
    'DynamicVoxelVFE_V2': DynamicVoxelVFE_V2,
}
