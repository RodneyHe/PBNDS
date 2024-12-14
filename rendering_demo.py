import torch
import torchvision.transforms.functional as TF
from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.ops import sample_points_from_meshes, interpolate_face_attributes
from pytorch3d.loss import chamfer_distance
from utils import CameraUtils

from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras
from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.mesh import MeshRendererWithFragments, MeshRasterizer, SoftPhongShader

from models.neural_renderer import NeuralRenderer

import matplotlib.pyplot as plt

# Parameter setting
# parser = configargparse.ArgumentParser()
# parser.add_argument('--config', is_config_file=True, help='config file path')
# parser.add_argument('--expname', type=str, help='experiment name')
# parser.add_argument('--basedir', type=str, default='./logs/', help='where to store ckpts and logs')
# parser.add_argument('--datadir', type=str, default='./data/llff/fern', help='input data directory')
# parser.add_argument('--device', type=str, default='gpu')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load meshes
lego_mesh_world = load_objs_as_meshes(["./mesh_data/lego/lego.obj"], device=device)

# Rasterizer setting
rasterizer_settings = RasterizationSettings(
    image_size=(512, 512), 
    blur_radius=0.0, 
    faces_per_pixel=1,
    bin_size=0
)

# Camera setting
R, T = look_at_view_transform(5, 20, 0)
cameras = PerspectiveCameras(R=R, T=T, device=device, in_ndc=False, image_size=rasterizer_settings.image_size)
                                                                                
# Instantiate rasterizer
rasterizer = MeshRasterizer(cameras=cameras, raster_settings=rasterizer_settings)

# Rendering
fragments = rasterizer(lego_mesh_world)