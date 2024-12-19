import torch
import torch.nn as nn

from models.neural_renderer import NeuralRenderer
from models.unet import UNet128

class PBNDR(nn.Module):
    def __init__(self, pretrained_weights=None):
        super().__init__()
        
        self.width, self.height = 128, 128
        
        neural_render = NeuralRenderer()
        shadow_estimator = UNet128(in_chns=6, out_chns=1)
        
        if pretrained_weights is not None:
            renderer_weights = torch.load(f'{pretrained_weights}/NeuralRenderer.pth')
            shadow_weights = torch.load(f'{pretrained_weights}/ShadowEstimator.pth')
            
            neural_render.load_state_dict({k: v for k, v in renderer_weights.items() if 'unet' not in k})
            shadow_estimator.load_state_dict(shadow_weights)
        
        self.neural_render = neural_render
        self.shadow_estimator = shadow_estimator
    
    def forward(self, render_buffer, mask, num_light_samples):
        
        rgb_pred = self.neural_render(render_buffer, num_light_samples)
        
        device = render_buffer['normal_gt'].device
        
        noraml = torch.zeros(128,128,3).to(device)
        noraml[mask] = render_buffer['normal_gt']
        
        rgb_rec = torch.zeros(128,128,3).to(device)
        rgb_rec[mask] = rgb_pred
        
        shadow_pred = self.shadow_estimator(torch.cat([rgb_rec[None], noraml[None]], dim=-1).permute(0,3,1,2))
        
        return rgb_pred, shadow_pred.permute(0,2,3,1)