import torch

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

from pytorch3d.renderer.mesh.textures import TexturesUV
from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings

def load_obj_files(self, mesh_paths, diffuse_images, roughness_images, metallic_images, device):
        
        verts_list = []
        faces_list = []
        verts_uvs_list = []
        faces_uvs_list = []
        maps_list = []
        for mesh_path, diffuse_image, roughness_image, metallic_image in zip(mesh_paths, diffuse_images, roughness_images, metallic_images):
            
            verts, faces, aux = load_obj(mesh_path, device=device)
            
            verts_list.append(verts.cuda())
            faces_list.append(faces.verts_idx.cuda())
            
            verts_uvs = aux.verts_uvs                                                                                                                   # [V, 2]
            faces_uvs = faces.textures_idx                                                                                                              # [F, 3]
            
            verts_uvs_list.append(verts_uvs.cuda())                                                                                                     # [N][V, 2]
            faces_uvs_list.append(faces_uvs.cuda())                                                                                                     # [N][F, 3]
            
            maps = torch.cat([diffuse_image.cuda(), roughness_image.cuda(), metallic_image.cuda()], -1)                                                 # [H, W, 5]
            maps_list.append(maps)                                                                                                                      # [N][H, W, 5]

        textures = TexturesUV(verts_uvs=verts_uvs_list, faces_uvs=faces_uvs_list, maps=maps_list)                                                       # [N, H, W, 5]
        meshes = Meshes(verts=verts_list, faces=faces_list, textures=textures)

        return meshes
