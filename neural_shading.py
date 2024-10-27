import torch

from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.ops import sample_points_from_meshes, interpolate_face_attributes
from pytorch3d.loss import chamfer_distance
from pytorch3d.renderer.cameras import look_at_view_transform, FoVPerspectiveCameras
from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.mesh import MeshRenderer, MeshRasterizer, SoftPhongShader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use an ico_sphere mesh and load a mesh from an .obj e.g. model.obj
sphere_mesh = ico_sphere(level=3)
mesh1 = load_objs_as_meshes(["./data/cow_mesh/cow.obj"], device=device)
mesh2 = load_objs_as_meshes(["./data/cow_mesh/cow.obj"], device=device)
mesh3 = load_objs_as_meshes(["./data/cow_mesh/cow.obj"], device=device)

obj1_world_coords = torch.tensor([0., 0., 2.]).to(device)
obj2_world_coords = torch.tensor([0., 0., 4.]).to(device)
obj3_world_coords = torch.tensor([0., 0., 6.]).to(device)

mesh1.verts_list()[0].add_(obj1_world_coords)
mesh2.verts_list()[0].add_(obj2_world_coords)
mesh3.verts_list()[0].add_(obj3_world_coords)

meshes = join_meshes_as_scene([mesh1, mesh2, mesh3])

# Initialize a camera.
# With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
# So we move the camera by 180 in the azimuth direction so it is facing the front of the cow.

R, T = look_at_view_transform(eye=torch.tensor([0., 1., -2.]).unsqueeze(0).to(device)) 
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
# the difference between naive and coarse-to-fine rasterization. 

raster_settings = RasterizationSettings(
    image_size=512, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)

# Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
# -z direction.
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

# Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
# interpolate the texture uv coordinates for each vertex, sample from a texture image and 
# apply the Phong lighting model
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights
    )
)

images, frag = renderer(meshes)
plt.figure(figsize=(5, 5))
plt.axis("off")
plt.imsave("./output/renderred_image.png", images[0, ..., :3].cpu().numpy())
