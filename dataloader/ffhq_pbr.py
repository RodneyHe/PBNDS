import os, math, json, random
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from utils.io import load_sdr, load_hdr

class FFHQPBR(Dataset):
    def __init__(self, data_path, eval_rate=0.1, mode=None):
        super().__init__()
        
        self.width, self.height = 128, 128
        
        self.data_path = data_path
        self.rgb_gt_path = self.data_path + f'/bgremoval'
        self.albedo_gt_path = self.data_path + f'/texture/albedo'
        self.normal_gt_path = self.data_path + f'/texture/normal'
        self.roughness_gt_path = self.data_path + f'/texture/roughness'
        self.specular_gt_path = self.data_path + f'/texture/specular'
        self.depth_gt_path = self.data_path + f'/texture/depth'
        self.light_gt_path = self.data_path + f'/hdri'
        self.fov_json_path = self.data_path + f'/pred_fov.json'
        
        self.gt_indices_list = self._get_meta_data_list()
        
        dataset_length = 20000
        train_num = round(dataset_length * (1 - eval_rate))
        
        with open(self.fov_json_path, 'r') as openfile:
            self.pred_fov_dict = json.load(openfile)
        
        if mode == 'train':
            self.data_list = self.gt_indices_list[:dataset_length][:train_num]
        elif mode == 'eval':
            self.data_list = self.gt_indices_list[:dataset_length][train_num:]
        elif mode == 'test':
            self.data_list = self.gt_indices_list[20000:20500]
    
    def _get_meta_data_list(self):
        
        rgb_subfolder_list = sorted(os.listdir(self.rgb_gt_path))
        
        gt_indices_list = []
        for sub_folder in rgb_subfolder_list:
            for name in sorted(os.listdir(os.path.join(self.rgb_gt_path, sub_folder))):
                gt_indices_list.append(os.path.join(name[:-4]))
        
        return gt_indices_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        
        file_index = self.data_list[index]
        subfolder = int(file_index) // 1000 * 1000
        
        rgb_gt = load_sdr(os.path.join(self.rgb_gt_path, f'{subfolder:05}/{file_index}.png'))
        normal_gt = load_sdr(os.path.join(self.normal_gt_path, f'{subfolder:05}/normal_{file_index}.png'))
        normal_gt = (normal_gt * 2 - 1.).float()
        albedo_gt = load_sdr(os.path.join(self.albedo_gt_path, f'{subfolder:05}/albedo_{file_index}.png'))
        roughness_gt = load_sdr(os.path.join(self.roughness_gt_path, f'{subfolder:05}/roughness_{file_index}.png'))
        specular_gt = load_sdr(os.path.join(self.specular_gt_path, f'{subfolder:05}/specular_{file_index}.png'))
        depth_gt = load_hdr(os.path.join(self.depth_gt_path, f'{subfolder:05}/depth_{file_index}.exr'))[...,0]
        hdri_gt = load_hdr(os.path.join(self.light_gt_path, f'{subfolder:05}/hdri_{file_index}.exr'), resize=False)
        mask_gt = (rgb_gt != 0)[...,0]
        
        # Get view pos from estimated fov
        pred_fov = self.pred_fov_dict[str(file_index)]
        
        pos_in_cam_gt = self.get_cam_pos(depth=depth_gt, width=self.width, height=self.height, fov=pred_fov)
        
        data_buffer = {
            'rgb_gt': rgb_gt,
            'normal_gt': normal_gt,
            'albedo_gt': albedo_gt,
            'roughness_gt': roughness_gt,
            'specular_gt': specular_gt,
            'depth_gt': depth_gt,
            'pos_in_cam_gt': pos_in_cam_gt,
            'mask_gt': mask_gt,
            'hdri_gt': hdri_gt,
            'file_index': str(file_index)
        }
        
        return data_buffer
    
    def get_cam_pos(self, depth, width, height, fov):
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

def get_dataloader(data_folder, eval_rate, random_seed, batch_size, shuffle, num_workers):
    
    # Fix random seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) 
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    def worker_init_fn(worker_id):
        np.random.seed(random_seed + worker_id)
    
    train_dataset = FFHQPBR(data_path=data_folder, 
                            eval_rate=eval_rate, 
                            mode='train')
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=shuffle, 
                              num_workers=num_workers, 
                              worker_init_fn=worker_init_fn,
                              pin_memory=True)
    
    eval_dataset = FFHQPBR(data_path=data_folder, 
                           eval_rate=eval_rate,
                           mode='eval')
    eval_loader = DataLoader(eval_dataset, 
                             batch_size=batch_size, 
                             shuffle=shuffle, 
                             num_workers=num_workers, 
                             worker_init_fn=worker_init_fn,
                             pin_memory=True)
    
    return train_loader, eval_loader
