from tqdm import tqdm

import torch
import torch.nn as nn

from models.positional_embedder import get_embedder
from models.neural_shader import FC

class NDS(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.width, self.height = 128, 128
        
        self.fourier_feature_transform, channels = get_embedder(8, 3)
        
        self.diffuse = FC(in_features=channels, 
                          out_features=256, 
                          hidden_features=[256, 256], 
                          activation='relu', 
                          last_activation=None)
        
        self.specular = FC(in_features=256 + 6, 
                           out_features=3, 
                           hidden_features=[256 // 2] * 2, 
                           activation='relu', 
                           last_activation=None)
        
    def forward(self, position, normal, out_dirs):
        
        position = torch.nn.functional.normalize(position, dim=-1)
        
        diffuse_shading_input = self.fourier_feature_transform(position)
        diffue_feature = self.diffuse(diffuse_shading_input)
        color = self.specular(torch.cat([diffue_feature, normal, out_dirs], dim=-1))
        
        return color.clamp(min=0.,max=1.)

class NDR(nn.Module):
    def __init__(self, nds_weights=None, fitting_num=3000):
        super().__init__()
        
        self.width, self.height = 128, 128
        
        self.fitting_num = fitting_num
        
        self.shader = NDS()
        
        if nds_weights is not None:
            self.shader.load_state_dict(nds_weights)
        
        # Optimizer
        self.adam_optimizer = torch.optim.Adam(params=self.shader.parameters(), lr=5e-5)
        
        # Loss function
        self.rec_loss = nn.L1Loss()
    
    def forward(self, position, normal, out_dirs):
        return self.shader(position, normal, out_dirs)
    
    def fit(self, position, normal, out_dirs, rgb_gt, save_path):
        
        pbar = tqdm(range(self.fitting_num), ncols=80)
        for _ in pbar:
            rgb_pred = self.shader(position, normal, out_dirs)
            
            rec_loss = self.rec_loss(rgb_pred, rgb_gt)
            
            if rec_loss.item() is not None:
                self.adam_optimizer.zero_grad()
                rec_loss.backward()
                self.adam_optimizer.step()
        
        torch.save(self.state_dict(), save_path)
        
        return rgb_pred
    