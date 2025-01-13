import torch
import torch.nn as nn

from models.neural_renderer import NeuralRenderer
from models.unet import UNet128

class PBNDR(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.width, self.height = 128, 128
        
        self.neural_render = NeuralRenderer()
        self.shadow_estimator = UNet128(in_chns=6, out_chns=1)
    
    def forward(self, render_buffer, mask, num_light_samples, shadowing=True):
        
        rgb_pred = self.neural_render(render_buffer, num_light_samples)
        
        device = render_buffer['normal_gt'].device
        
        noraml = torch.zeros(128,128,3).to(device)
        noraml[mask] = render_buffer['normal_gt']
        
        rgb_rec = torch.zeros(128,128,3).to(device)
        rgb_rec[mask] = rgb_pred
        
        if shadowing:
            shadow_pred = self.shadow_estimator(torch.cat([rgb_rec[None], noraml[None]], dim=-1).permute(0,3,1,2))
            return rgb_pred, shadow_pred.permute(0,2,3,1)
        
        return rgb_pred