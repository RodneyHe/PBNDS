import torch
import torch.nn as nn

from utils.sampler import Sampler

from .neural_shader import NeuralShader

class NeuralRenderer(nn.Module):
    def __init__(self):
        super().__init__()
        self.shader = NeuralShader()
        env_width, env_height = 64, 32
        
        # Initialze local environment map 
        # Azimuth range (-pi - pi)
        Az = ((torch.arange(env_width) + 0.5) / env_width - 0.5) * 2 * torch.pi
        
        # Elevation range (0 - 0.5 pi)
        El = ((torch.arange(env_height) + 0.5) / env_height) * torch.pi * 0.5
        
        El, Az = torch.meshgrid(El, Az, indexing='ij')
        
        Az = Az[:, :, None]
        El = El[:, :, None]
        
        # X:left; Y: up; Z: out of screen.
        lx = torch.cos(Az) * torch.cos(El)
        ly = torch.sin(El)
        lz = torch.sin(Az) * torch.cos(El)
        
        ls = torch.cat([lx, ly, lz], dim=-1).reshape(-1, 3)
        self.ls = nn.Parameter(ls, requires_grad=False)
        
        cam_pos = torch.tensor([0., 0., 0.])[None, None, :]
        self.cam_pos = nn.Parameter(cam_pos, requires_grad=False)
        
        self.sampler = Sampler()

    def forward(self, render_buffer, num_light_samples):
        
        pos_in_cam_gt = render_buffer['pos_in_cam_gt']                                                                                 # [B,H,W]
        hdri_gt = render_buffer['hdri_gt']                                                                                     # [B,env_h,env_w,3]
        
        # Sampling the HDRi environment map, getting sampled light and inbound direction
        sampled_hdri_map, sampled_direction = self.sampler.uniform_sampling(hdri_map=hdri_gt, num_samples=num_light_samples)
        
        # Calculate outbound direction
        in_dirs = sampled_direction.repeat(pos_in_cam_gt.shape[0],1,1)                                                               # [S,N,3]
        out_dirs = (self.cam_pos - pos_in_cam_gt.unsqueeze(1))
        out_dirs = nn.functional.normalize(out_dirs, dim=-1)                                                                    # [S,N,3]
        
        shading_input = {
            'normal': render_buffer['normal_gt'].unsqueeze(1).broadcast_to(in_dirs.shape),                                      # [S,N,3]
            'albedo': render_buffer['albedo_gt'].unsqueeze(1).broadcast_to(in_dirs.shape),                                      # [S,N,3]
            'roughness': render_buffer['roughness_gt'].unsqueeze(1)[...,None].broadcast_to(*in_dirs.shape[:-1],1),              # [S,N,1]
            'specular': render_buffer['specular_gt'].unsqueeze(1)[...,None].broadcast_to(*in_dirs.shape[:-1],1),                # [S,N,1]
            'in_dirs': in_dirs,                                                                                                 # [S,N,3]
            'out_dirs': out_dirs.broadcast_to(in_dirs.shape),                                                                   # [S,N,3]
            'hdri_samples': sampled_hdri_map.broadcast_to(in_dirs.shape)                                                        # [S,N,3]
        }
        
        masked_rgb_pixels = self.shader(**shading_input)

        return masked_rgb_pixels
    
    # Utility function
    def split_model_inputs(self, input, total_pixels, split_size):
        '''
        Split the input to fit Cuda memory for large resolution.
        Can decrease the value of split_num in case of cuda out of memory error.
        '''
        split_size = split_size                                                                                                            # [S]
        split_input = []
        split_indexes = torch.split(torch.arange(total_pixels).cuda(), split_size, dim=0)
        for indexes in split_indexes:
            data = {}
            data['normal'] = torch.index_select(input['normal'], 1, indexes)
            data['albedo'] = torch.index_select(input['albedo'], 1, indexes)
            data['roughness'] = torch.index_select(input['roughness'], 1, indexes)
            data['specular'] = torch.index_select(input['specular'], 1, indexes)
            data['in_dirs'] = torch.index_select(input['in_dirs'], 1, indexes)
            data['out_dirs'] = torch.index_select(input['out_dirs'], 1, indexes)
            data['hdri_samples'] = torch.index_select(input['hdri_samples'], 1, indexes)
            split_input.append(data)
            
        return split_input
    
    def save_model(self, weights_dir, reason=""):
        torch.save(self.state_dict(), weights_dir + f"/{self.__class__.__name__ + reason}.pth")
