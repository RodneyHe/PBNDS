import os, random, math
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import numpy as np
import cv2 as cv

import torch
from torch.utils.data import Dataset

class FFHQPBR(Dataset):
    def __init__(self, data_path, eval_rate=0.1, random_seed=0, mode=None, stage='1'):
        super().__init__()
        
        self.width, self.height = 128, 128
        self.stage = stage
        self.mode = mode
        torch.manual_seed(random_seed)
        
        self.data_path = data_path
        self.rgb_gt_path = self.data_path + f'/bgremoval'
        self.albedo_gt_path = self.data_path + f'/texture/albedo'
        self.normal_gt_path = self.data_path + f'/texture/normal'
        self.roughness_gt_path = self.data_path + f'/texture/roughness'
        self.specular_gt_path = self.data_path + f'/texture/specular'
        self.depth_gt_path = self.data_path + f'/texture/depth'
        self.light_gt_path = self.data_path + f'/hdri'
        
        self.gt_indices_list = self._get_meta_data_list()[:20000]
        
        self.dataset_length = len(self.gt_indices_list)
        train_num = round(self.dataset_length * (1 - eval_rate))
        
        if mode == 'train':
            self.train_list = self.gt_indices_list[:train_num]
        elif mode == 'eval':
            self.eval_list = self.gt_indices_list[train_num:]
    
    def _get_meta_data_list(self):
        
        rgb_subfolder_list = sorted(os.listdir(self.rgb_gt_path))
        
        gt_indices_list = []
        for sub_folder in rgb_subfolder_list:
            for name in sorted(os.listdir(os.path.join(self.rgb_gt_path, sub_folder))):
                gt_indices_list.append(os.path.join(name[:-4]))
        
        return gt_indices_list
    
    def __len__(self):
        return self.dataset_length
    
    def __getitem__(self, index):
        
        file_index = self.gt_indices_list[index]
        subfolder = int(file_index) // 1000 * 1000
        
        rgb_gt = self.load_sdr(os.path.join(self.rgb_gt_path, f'{subfolder:05}/{file_index}.png'))
        rgb_gt = rgb_gt / 255.
        normal_gt = self.load_sdr(os.path.join(self.normal_gt_path, f'{subfolder:05}/normal_{file_index}.png'))
        normal_gt = ((normal_gt / 255.) * 2 - 1.).to(torch.float32)
        albedo_gt = self.load_sdr(os.path.join(self.albedo_gt_path, f'{subfolder:05}/albedo_{file_index}.png'))
        albedo_gt = albedo_gt / 255.
        roughness_gt = self.load_sdr(os.path.join(self.roughness_gt_path, f'{subfolder:05}/roughness_{file_index}.png'))
        roughness_gt = roughness_gt / 255.
        specular_gt = self.load_sdr(os.path.join(self.specular_gt_path, f'{subfolder:05}/specular_{file_index}.png'))
        specular_gt = specular_gt / 255.
        depth_gt = self.load_hdr(os.path.join(self.depth_gt_path, f'{subfolder:05}/depth_{file_index}.exr'), \
                                 resize=True)[...,0]
        mask_gt = (rgb_gt != 0)[...,0]
        hdri_gt = self.load_hdr(os.path.join(self.light_gt_path, f'{subfolder:05}/hdri_{file_index}.exr'))
        
        # Get view pos via fixed fov (50 degree in this case) and depth ground truth
        view_pos_gt = self.get_view_pos(depth=depth_gt, width=self.width, height=self.height, fov=50)

        data_buffer = {
            'rgb_gt': rgb_gt,
            'normal_gt': normal_gt,
            'albedo_gt': albedo_gt,
            'roughness_gt': roughness_gt,
            'specular_gt': specular_gt,
            'view_pos_gt': view_pos_gt,
            'hdri_gt': hdri_gt,
            'mask_gt': mask_gt
        }
        
        return data_buffer
    
    def get_view_pos(self, depth, width, height, fov):
        fovx = math.radians(fov)
        fovy = 2 * math.atan(math.tan(fovx / 2) / (width / height))
        vpos = torch.zeros(height, width, 3)
        Y = 1 - (torch.arange(height) + 0.5) / height
        Y = Y * 2 - 1
        X = (torch.arange(width) + 0.5) / width
        X = X * 2 - 1
        Y, X = torch.meshgrid(Y, X, indexing='ij')
        vpos[..., 0] = depth * X * math.tan(fovx / 2)
        vpos[..., 1] = depth * Y * math.tan(fovy / 2)
        vpos[..., 2] = -depth
        return vpos
    
    def load_sdr(self, image_name):
        image = cv.imread(image_name, cv.IMREAD_UNCHANGED)
        
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                alpha_channel = image[...,3]
                bgr_channels = image[...,:3]
                rgb_channels = cv.cvtColor(bgr_channels, cv.COLOR_BGR2RGB)
                
                # White Background Image
                background_image = np.zeros_like(rgb_channels, dtype=np.uint8)
                
                # Alpha factor
                alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) / 255.
                alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)

                # Transparent Image Rendered on White Background
                base = rgb_channels * alpha_factor
                background = background_image * (1 - alpha_factor)
                image = base + background
            else:
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        image = cv.resize(image, (self.width, self.height), interpolation=cv.INTER_NEAREST)
        
        return torch.from_numpy(image)
    
    def load_hdr(self, image_name, resize=False, to_ldr=False):
        image = cv.imread(image_name, -1)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        if resize:
            image = cv.resize(image, (self.width, self.height), interpolation=cv.INTER_NEAREST)
        
        if to_ldr:
            image = image.clip(0, 1) ** (1 / 2.2)
        
        return torch.from_numpy(image)
