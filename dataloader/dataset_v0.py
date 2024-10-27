import json, torch, os, random
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
                 num_pixel_samples,
                 use_brdf_gt = False,
                 mode='train'):
        super().__init__()
        
        assert os.path.exists(data_folder), 'Data directory is empty'
        
        self.data_folder = data_folder
        self.mode = mode
        self.num_pixel_samples = num_pixel_samples
        
        if use_brdf_gt:
            self.postfix = 'png'
        else:
            self.postfix = 'exr'

        scene_list = os.listdir(data_folder)
        random.shuffle(scene_list)
        
        camera = PerspectiveCameras(in_ndc=False, image_size=[(1024, 1024)])
        rasterizer_settings = RasterizationSettings(image_size=1024)
        self.rasterizer = MeshRasterizer(raster_settings=rasterizer_settings, cameras=camera)
        
        self.P2B_transform  = torch.tensor([[1, 0, 0],
                                            [0, 0, 1],
                                            [0,-1, 0]], dtype=torch.float32)                             # [3, 3]
        
        # train data processing
        if mode == 'train':
            
            print(f'Loading training data from {data_folder}, number of training data is {len(scene_list) - validation_amount}.')
            
            train_scene_list = scene_list[:-validation_amount]
            self._load_and_prepare_training_data(train_scene_list)
        
        # validatio data processing
        if mode == 'validation':
            
            print(f'Loading validation data from {data_folder}, number of validation data is {validation_amount}.')
            
            validate_scene_list = scene_list[-validation_amount:]
            self._load_and_prepare_validation_data(validate_scene_list)
    
    def _load_and_prepare_training_data(self, train_list):
        
        self._load_data_from_scene(train_list)
        
        self.all_scene_index = torch.tensor(self.all_scene_index)
        self.all_image_index = torch.tensor(self.all_image_index)
        self.all_num_training_pixels = torch.tensor(self.all_num_training_pixels)
        
        self.num_scene_index = self.all_scene_index.shape[0]
        
        self._random_permute_all_scene_indices()
        self._random_permute_all_training_pixels()
    
    def _load_and_prepare_validation_data(self, validate_list):
        
        self._load_data_from_scene(validate_list)
        
        self.all_scene_index = torch.tensor(self.all_scene_index)
        self.all_image_index = torch.tensor(self.all_image_index)
        self.all_num_training_pixels = torch.tensor(self.all_num_training_pixels)
        
        self.num_scene_index = self.all_scene_index.shape[0]
        
        self._random_permute_all_scene_indices()
        self._random_permute_all_training_pixels()
    
    def _load_data_from_scene(self, scene_list):
        
        self.all_scene_index = []
        self.all_image_index = []
        self.all_num_training_pixels = []
        
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
            scene_index_list = []
            image_index_list = []
            num_training_pxiels = []
            
            for list_index, (image_index, camera_info) in enumerate(image_list.items()):
                
                image_index = int(image_index)
                
                W, H = camera_info["size"]
                
                scene_index_list.append(scene_index)
                image_index_list.append(image_index)
                
                num_training_pxiels.append(W * H)

            self.all_scene_index.extend(scene_index_list)
            self.all_image_index.extend(image_index_list)
            self.all_num_training_pixels.extend(num_training_pxiels)
    
    def _random_permute_all_scene_indices(self):
        
        rand_scene_index = torch.randperm(self.num_scene_index)
        self.rand_scene_index = self.all_scene_index[rand_scene_index]
        self.rand_image_index = self.all_image_index[rand_scene_index]
        self.rand_num_training_pixels = self.all_num_training_pixels[rand_scene_index]
        
        self.scene_index_in_rand = 0
        self.image_index_in_rand = 0
    
    def _random_permute_all_training_pixels(self):
        
        self.num_training_pixels = self.rand_num_training_pixels[self.scene_index_in_rand]
        self.rand_pixel_indices = torch.randperm(self.num_training_pixels)
        
        self.pixel_index_in_rand = 0

    def __len__(self):
        return 50000

    def __getitem__(self, index):
        
        scene_index = self.rand_scene_index[self.scene_index_in_rand]
        image_index = self.rand_image_index[self.image_index_in_rand]
        
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
        json_data = self.all_scene_json_data[scene_index.item()]
        camera_info = json_data["camera_track_map"]["images"][str(image_index.item())]
        
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
        
        textures = mesh.textures.sample_textures(fragments).squeeze(3).reshape(-1, 5)                                           # [HW, 5]
        
        # get position map
        mesh_verts_world = mesh.verts_packed()                                                                                  # [1, V, 3]
        mesh_faces_normal = mesh.faces_normals_packed().unsqueeze(1).repeat(1, 3, 1)                                            # [1, F, 3, 3]
        
        mesh_faces_indices = mesh.faces_packed()                                                                                # [1, F, 3]
        mesh_faces_verts_world = mesh_verts_world[mesh_faces_indices]                                                           # [1, F, 3]
        
        positions = interpolate_face_attributes(fragments.pix_to_face, 
                                                fragments.bary_coords, 
                                                mesh_faces_verts_world).reshape(-1, 3)                                          # [HW, 3]
        
        normals = interpolate_face_attributes(fragments.pix_to_face, 
                                              fragments.bary_coords, 
                                              mesh_faces_normal).reshape(-1, 3)                                                 # [HW, 3]
        
        camera_locations = self.rasterizer.cameras.get_camera_center().broadcast_to(*textures.shape[:-1], 3).reshape(-1, 3)     # [HW, 3]
        
        masks = (positions != 0).sum(-1) > 0                                                                                    # [HW, 3]
        
        ray_directions = positions - camera_locations
        ray_directions = torch.nn.functional.normalize(ray_directions, dim=1)                                                   # [HW, 3]
        
        positions = torch.matmul(positions, self.P2B_transform).squeeze(-1)
        normals = torch.matmul(normals, self.P2B_transform).squeeze(-1)
        ray_directions = torch.matmul(ray_directions, self.P2B_transform).squeeze(-1)
        view_directions = -ray_directions
        
        # Load ground truth rgb image
        rgb_gt_image = io.load_rgb_image(self.data_folder + f'/scene_{scene_index}/inputs/images/{image_index}.{self.postfix}')
        rgb_gt_image = torch.from_numpy(rgb_gt_image).reshape(-1, 3)                                                             # [HW, 3]
        
        # Load neilf weights
        outputs_folder = self.data_folder + f'/scene_{scene_index}/outputs'
        timestamp_list = os.listdir(outputs_folder)
        timestamp_list.sort(reverse=True, key=lambda x:x)
        neilf_weights_path = outputs_folder + f'/{timestamp_list[0]}/checkpoints/ModelParameters/latest.pth'
        
        if self.mode == "train":
            
            # Prepare training sample data
            # Permute training pxiels
            positions = positions[self.rand_pixel_indices]
            normals = normals[self.rand_pixel_indices]
            view_directions = view_directions[self.rand_pixel_indices]
            textures = textures[self.rand_pixel_indices]
            masks = masks[self.rand_pixel_indices]
            
            start_pixel_index = self.pixel_index_in_rand
            end_pixel_index = self.pixel_index_in_rand + self.num_pixel_samples
            
            # Permute all pxiels only after traversing all
            self.pixel_index_in_rand += self.num_pixel_samples
            if self.pixel_index_in_rand > self.num_training_pixels.item():
                
                self.scene_index_in_rand += 1
                self.image_index_in_rand += 1
                if self.scene_index_in_rand > self.num_scene_index:
                    self._random_permute_all_scene_indices()
                
                self._random_permute_all_training_pixels()
                
                start_pixel_index = self.pixel_index_in_rand
                end_pixel_index = self.pixel_index_in_rand + self.num_pixel_samples
            
            batch_positions = positions[start_pixel_index:end_pixel_index, :]
            batch_normals = normals[start_pixel_index:end_pixel_index, :]
            batch_view_directions = view_directions[start_pixel_index:end_pixel_index, :]
            batch_textures = textures[start_pixel_index:end_pixel_index, :]
            batch_masks = masks[start_pixel_index:end_pixel_index]
            batch_rgb_gt = rgb_gt_image[start_pixel_index:end_pixel_index, :]
        
            sample = {
                'positions': batch_positions,
                'normals': batch_normals,
                'view_dirs': batch_view_directions,
                'textures': batch_textures,
                'masks': batch_masks,
                'neilf_weights_path': neilf_weights_path
            }
        
            ground_truth = {
                'rgb_pxiels': batch_rgb_gt
            }
        
        elif self.mode == "validation":
            
            self.scene_index_in_rand += 1
            self.image_index_in_rand += 1
            
            sample = {
                'positions': positions,
                'normals': normals,
                'view_dirs': view_directions,
                'textures': textures,
                'masks': masks,
                'neilf_weights_path': neilf_weights_path
            }
            
            ground_truth = {
                'rgb_pxiels': rgb_gt_image
            }

        return sample, ground_truth
