import sys, os, math, json
sys.path.append('..')
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

from utils.io import load_sdr, load_hdr
from utils.sampler import Sampler

import torch
import torch.nn as nn
import torchvision.transforms.functional as tvf

from models.neural_renderer import NeuralRenderer
from models.unet import UNet128

class PBNDR(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.width, self.height = 128, 128
        
        self.sampler = Sampler()
        
        self.neural_renderer = NeuralRenderer()
        self.shadow_estimator = UNet128(in_chns=6, out_chns=1)
        
        # Load weights
        self.neural_renderer.load_state_dict(torch.load('weights/NeuralRenderer.pth'))
        self.shadow_estimator.load_state_dict(torch.load('weights/ShadowEstimator.pth'))
        
        self.cam_pos = torch.tensor([0., 0., 0.])[None, None, :]
    
    def forward(self, render_buffer, shadowing=True):
        
        pos_in_cam_gt = render_buffer['pos_in_cam_gt']
        hdri_gt = render_buffer['hdri_gt']
        mask_gt = render_buffer['mask_gt']
        
        # Sampling the HDRi environment map, getting sampled light and inbound direction
        sampled_hdri_map, sampled_direction = self.sampler.uniform_sampling(hdri_map=hdri_gt, num_samples=1024)
        
        # Calculate outbound direction
        in_dirs = sampled_direction.repeat(pos_in_cam_gt.shape[0],1,1)                                                               # [S,N,3]
        out_dirs = (self.cam_pos - pos_in_cam_gt.unsqueeze(1))
        out_dirs = nn.functional.normalize(out_dirs, dim=-1)
        
        shading_input = {
            'normal': render_buffer['normal_gt'].unsqueeze(1).broadcast_to(in_dirs.shape), 
            'albedo': render_buffer['albedo_gt'].unsqueeze(1).broadcast_to(in_dirs.shape), 
            'roughness': render_buffer['roughness_gt'].unsqueeze(1)[...,None].broadcast_to(*in_dirs.shape[:-1],1), 
            'specular': render_buffer['specular_gt'].unsqueeze(1)[...,None].broadcast_to(*in_dirs.shape[:-1],1), 
            'in_dirs': in_dirs, 
            'out_dirs': out_dirs.broadcast_to(in_dirs.shape), 
            'hdri_samples': sampled_hdri_map.broadcast_to(in_dirs.shape)
        }
        
        rgb_pred = self.neural_renderer.shader(**shading_input)
        
        device = render_buffer['normal_gt'].device
        
        noraml = torch.zeros(128,128,3).to(device)
        noraml[mask_gt] = render_buffer['normal_gt']
        
        rgb_rec = torch.zeros(128,128,3).to(device)
        rgb_rec[mask_gt] = rgb_pred
        
        if shadowing:
            shadow_pred = self.shadow_estimator(torch.cat([rgb_rec[None], noraml[None]], dim=-1).permute(0,3,1,2))
            return rgb_rec, shadow_pred.permute(0,2,3,1)[0].repeat(1,1,3)
        
        return rgb_rec

def get_cam_pos(depth, width, height, fov):
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

if __name__ == '__main__':
    
    # Load model
    pbndr = PBNDR()
    
    # Load data
    index = '43982'
    rgb_gt = load_sdr(f'gbuffer/rgb_{index}.png')
    normal_gt = load_sdr(f'gbuffer/normal_{index}.png')
    normal_gt = (normal_gt * 2 - 1).float()
    albedo_gt = load_sdr(f'gbuffer/albedo_{index}.png')
    roughness_gt = load_sdr(f'gbuffer/roughness_{index}.png')
    specular_gt = load_sdr(f'gbuffer/specular_{index}.png')
    depth_gt = load_hdr(f'gbuffer/depth_{index}.exr')[...,0]
    hdri_gt = load_hdr(f'hdri/05.exr', resize=False)
    mask_gt = (rgb_gt != 0)[...,0]
    
    with open('pred_fov.json', 'r') as openfile:
        pred_fov_dict = json.load(openfile)
    
    pred_fov = pred_fov_dict[index]
    pos_in_cam_gt = get_cam_pos(depth=depth_gt, width=pbndr.width, height=pbndr.height, fov=pred_fov)
    
    render_buffer = {
        'normal_gt': normal_gt[mask_gt],
        'albedo_gt': albedo_gt[mask_gt],
        'roughness_gt': roughness_gt[mask_gt],
        'specular_gt': specular_gt[mask_gt],
        'depth_gt': depth_gt[mask_gt],
        'pos_in_cam_gt': pos_in_cam_gt[mask_gt],
        'hdri_gt': hdri_gt[None],
        'mask_gt': mask_gt
    }
    
    with torch.no_grad():
        rendering_image, shadow_map = pbndr(render_buffer, shadowing=True)
    
    tvf.to_pil_image((rendering_image * shadow_map).permute(2,0,1)).save(f'pred_rgb_{index}_shadowed.png')
    #tvf.to_pil_image(rendering_image.permute(2,0,1)).save(f'pred_rgb_{index}_unshadowed.png')
    #tvf.to_pil_image(shadow_map.permute(2,0,1)).save(f'shadow_map_{index}.png')
    