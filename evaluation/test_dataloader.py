import os, math, json, random
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from utils.io import load_sdr, load_hdr

class TESTDATA(Dataset):
    def __init__(self, data_folder):
        super().__init__()
        
        self.width, self.height = 128, 128
        
        self.data_folder = data_folder
        self.fov_json_path = f'data/pred_fov.json'
        
        with open(self.fov_json_path, 'r') as openfile:
            self.pred_fov_dict = json.load(openfile)
        
        data_list = sorted(os.listdir(f'data/{data_folder}'))
        
        self.data_index_list = []
        for data_name in data_list:
            if data_name[:3] == 'rgb':
                self.data_index_list.append(data_name[4:9])
    
    def __len__(self):
        return len(self.data_index_list)
    
    def __getitem__(self, index):
        
        file_index = self.data_index_list[index]
        
        rgb_gt = load_sdr(os.path.join(f'data/{self.data_folder}/rgb_{file_index}.png'))
        normal_gt = load_sdr(os.path.join(f'data/{self.data_folder}/normal_{file_index}.png'))
        normal_gt = (normal_gt * 2 - 1.).float()
        albedo_gt = load_sdr(os.path.join(f'data/{self.data_folder}/albedo_{file_index}.png'))
        roughness_gt = load_sdr(os.path.join(f'data/{self.data_folder}/roughness_{file_index}.png'))
        specular_gt = load_sdr(os.path.join(f'data/{self.data_folder}/specular_{file_index}.png'))
        depth_gt = load_hdr(os.path.join(f'data/{self.data_folder}/depth_{file_index}.exr'))[...,0]
        hdri_gt = load_hdr(os.path.join(f'data/{self.data_folder}/hdri_{file_index}.exr'), resize=False)
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

def get_dataloader(data_folder):
    
    test_dataset = TESTDATA(data_folder)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
    
    return test_loader
