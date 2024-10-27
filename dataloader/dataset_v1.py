import json, torch, os, random
from pathlib import Path
import numpy as np

from torch.utils.data import Dataset
import torchvision.transforms.functional as TFunc

from pytorch3d.io import load_obj
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.textures import TexturesUV

from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.mesh import MeshRasterizer, RasterizationSettings

from utils import transforms, io

class NDSDataset(Dataset):
    
    def __init__(self,
                 data_folder,
                 validation_amount,
                 num_sample_pixels,
                 use_ldr_gt = False,
                 mode='train'):
        super().__init__()
        
        assert os.path.exists(data_folder), 'Data directory is empty'
        
        '''
        number_pixel_samples is as same as the chunk size
        '''
        
        self.data_folder = data_folder
        self.num_sample_pixels = num_sample_pixels
        self.mode = mode
        self.cache_folder = Path('./data_cache')
        
        if use_ldr_gt:
            self.postfix = 'png'
        else:
            self.postfix = 'exr'
        
        camera = PerspectiveCameras(in_ndc=False, image_size=[(1024, 1024)])
        rasterizer_settings = RasterizationSettings(image_size=1024)
        self.rasterizer = MeshRasterizer(raster_settings=rasterizer_settings, cameras=camera)
        
        self.P2B_transform  = torch.tensor([[1, 0, 0],
                                            [0, 0, 1],
                                            [0,-1, 0]], dtype=torch.float32)                             # [3, 3]
        
        data_list = os.listdir(data_folder)
        
        # train data processing
        if mode == 'train':
            
            print(f'Loading training data from {data_folder}, number of training data is {len(data_list) - validation_amount}.')
            
            self.scene_index_list, self.image_index_list = self._get_meta_data_list(data_list[:-validation_amount])

        # validatio data processing
        if mode == 'validation':
            
            print(f'Loading validation data from {data_folder}, number of validation data is {validation_amount}.')
            
            self.scene_index_list, self.image_index_list = self._get_meta_data_list(data_list[-validation_amount:])
            
            # random.shuffle(val_scene_index_list)
            # random.shuffle(val_image_index_list)
    
    def _get_meta_data_list(self, scene_list):
        
        scene_index_list = []
        image_index_list = []
        
        self.all_scene_json_data = dict()
        
        for scene in scene_list:
            scene_index = int(scene.split('_')[-1])
            
            # Load json data
            json_file_path = self.data_folder + f'/{scene}/inputs/synthetic_scene.json'
        
            with open(json_file_path) as f:
                json_data = json.load(f)
            
            self.all_scene_json_data[scene_index] = json_data
            
            # Prepare meta data
            image_list = json_data['camera_track_map']['images']
            
            for image_index, camera_info in image_list.items():
                
                scene_index_list.append(scene_index)
                image_index_list.append(image_index)
        
        return scene_index_list, image_index_list
    
    def _grouped(self, iterable, n):
        """
        s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
        """
        return zip(*[iter(iterable)]*n)

    def __len__(self):
        return len(self.image_index_list)

    def __getitem__(self, index):
        
        scene_index = self.scene_index_list[index]
        image_index = self.image_index_list[index]
        
        # load mesh data
        verts, faces, aux = load_obj(self.data_folder + f'/scene_{scene_index}/inputs/model/mesh.obj')
        
        # Load baked texture images
        diffuse_image = io.load_rgb_image(self.data_folder + f'/scene_{scene_index}/inputs/model/diffuse.{self.postfix}')
        diffuse_image = torch.from_numpy(diffuse_image)
        
        roughness_image = io.load_rgb_image(self.data_folder + f'/scene_{scene_index}/inputs/model/roughness.{self.postfix}')[..., 0]
        roughness_image = torch.from_numpy(roughness_image).unsqueeze(-1)
        
        metallic_image = io.load_rgb_image(self.data_folder + f'/scene_{scene_index}/inputs/model/metallic.{self.postfix}')[..., 0]
        metallic_image = torch.from_numpy(metallic_image).unsqueeze(-1)
        
        verts_uvs = aux.verts_uvs                                                                   # (V, 2)
        faces_uvs = faces.textures_idx                                                              # (F, 3)
        texture = torch.cat([diffuse_image, roughness_image, metallic_image], dim=-1)[None]

        mesh = Meshes(verts=[verts], faces=[faces.verts_idx], 
                      textures=TexturesUV(verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=texture))
        
        # Load materials and positions
        json_data = self.all_scene_json_data[scene_index]
        camera_info = json_data["camera_track_map"]["images"][str(image_index)]
        
        # flg == 2 stands for valid camera
        if camera_info['flg'] == 2:
            intrinsic = np.zeros((4, 4))
            intrinsic[0, 0] = camera_info['camera']['intrinsic']['focal'][0]
            intrinsic[1, 1] = camera_info['camera']['intrinsic']['focal'][1]
            intrinsic[0, 2] = camera_info['camera']['intrinsic']['ppt'][0]
            intrinsic[1, 2] = camera_info['camera']['intrinsic']['ppt'][1]
            
            # swap 3rd and 4th row for the consistency of pytorch3d
            intrinsic[2, 3] = intrinsic[3, 2] = 1
            intrinsic = torch.from_numpy(intrinsic).float()
            
            extrinsic = np.array(camera_info['camera']['extrinsic']).reshape(4, 4)                                          # [1, 4, 4]
            extrinsic = torch.from_numpy(extrinsic).float()
            
            # Transform blender extrinsic matrix to pytorch3d convention
            R, T, RT_4x4 = transforms.B2P(extrinsic.clone())
        
        self.rasterizer.cameras.K = intrinsic[None]
        self.rasterizer.cameras.R = R[None]
        self.rasterizer.cameras.T = T[None]
        
        with torch.no_grad():
            fragments = self.rasterizer(mesh)
        
        textures = mesh.textures.sample_textures(fragments).squeeze(3).reshape(1, -1, 5)                                        # [1, HW, 5]
        
        # get position map
        mesh_verts_world = mesh.verts_packed()                                                                                  # [1, V, 3]
        mesh_faces_normal = mesh.faces_normals_packed().unsqueeze(1).repeat(1, 3, 1)                                            # [1, F, 3, 3]
        
        mesh_faces_indices = mesh.faces_packed()                                                                                # [1, F, 3]
        mesh_faces_verts_world = mesh_verts_world[mesh_faces_indices]                                                           # [1, F, 3]
        
        positions = interpolate_face_attributes(fragments.pix_to_face, 
                                                fragments.bary_coords, 
                                                mesh_faces_verts_world).reshape(1, -1, 3)                                       # [1, HW, 3]
        
        normals = interpolate_face_attributes(fragments.pix_to_face, 
                                              fragments.bary_coords, 
                                              mesh_faces_normal).reshape(1, -1, 3)                                              # [1, HW, 3]
        
        camera_locations = self.rasterizer.cameras.get_camera_center()
        camera_locations = camera_locations.broadcast_to(*textures.shape[:-1], 3).reshape(1, -1, 3)                             # [1, HW, 3]
        
        masks = (positions != 0).sum(-1) > 0                                                                                    # [1, HW]
        
        ray_directions = positions - camera_locations
        ray_directions = torch.nn.functional.normalize(ray_directions, dim=1)                                                   # [1, HW, 3]
        
        positions = torch.matmul(positions, self.P2B_transform).squeeze(-1)
        normals = torch.matmul(normals, self.P2B_transform).squeeze(-1)
        ray_directions = torch.matmul(ray_directions, self.P2B_transform).squeeze(-1)
        view_directions = -ray_directions
        
        # Load ground truth rgb image
        rgb_gt_image_path = self.data_folder + f'/scene_{scene_index}/inputs/images/{image_index}.{self.postfix}'
        rgb_gt_image = io.load_rgb_image(rgb_gt_image_path)
        rgb_gt_image = torch.from_numpy(rgb_gt_image).reshape(1, -1, 3)                                                         # [1, HW, 3]
        
        # Load neilf weights
        outputs_folder = self.data_folder + f'/scene_{scene_index}/outputs'
        timestamp_list = os.listdir(outputs_folder)
        timestamp_list.sort(reverse=True, key=lambda x:x)
        neilf_weights_path = outputs_folder + f'/{timestamp_list[0]}/checkpoints/ModelParameters/latest.pth'
        
        if self.mode == "train":
            
            pixel_size = positions[masks].size(0)
            rand_index = torch.randperm(pixel_size)
            
            # Randomly sample pixels
            sampled_valid_positions = positions[masks][rand_index][:self.num_sample_pixels]
            sampled_valid_normals = normals[masks][rand_index][:self.num_sample_pixels]
            sampled_valid_view_directions = view_directions[masks][rand_index][:self.num_sample_pixels]
            sampled_valid_textures = textures[masks][rand_index][:self.num_sample_pixels]
            
            sampled_valid_rgb_gt = rgb_gt_image[masks][rand_index][:self.num_sample_pixels]
        
            sample = {
                'positions': sampled_valid_positions,
                'normals': sampled_valid_normals,
                'view_dirs': sampled_valid_view_directions,
                'textures': sampled_valid_textures,
                'neilf_weights_path': neilf_weights_path
            }
        
            ground_truth = {
                'rgb_pxiels': sampled_valid_rgb_gt
            }
        
        elif self.mode == "validation":
            
            sample = {
                'positions': positions.reshape(-1, 3),
                'normals': normals.reshape(-1, 3),
                'view_dirs': view_directions.reshape(-1, 3),
                'textures': textures.reshape(-1, 5),
                'masks': masks.reshape(-1, 1),
                'neilf_weights_path': neilf_weights_path
            }
            
            ground_truth = {
                'rgb_pxiels': rgb_gt_image.reshape(-1, 3)
            }

        return sample, ground_truth
