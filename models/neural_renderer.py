import math
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np

import torch
import torch.nn as nn

import torchvision.transforms.functional as tvf

from .neural_shader import NeuralShader

class NeuralRenderer(nn.Module):
    def __init__(self):
        super().__init__()
        self.shader = NeuralShader()
        self.sampling_method = 'uniform'            # importance or uniform
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

    def forward(self, render_buffer, num_samples=512, train=True):
        
        view_pos = render_buffer['view_pos_gt']                                                                                 # [B,H,W]
        hdri_map = render_buffer['hdri_gt']                                                                                     # [B,env_h,env_w,3]
        
        # Sampling the HDRi environment map
        sampled_hdri_map, sampled_direction = self.uniform_sampling(hdri_map=hdri_map, num_samples=num_samples)
        
        # Calculate outbound direction
        in_dirs = sampled_direction.repeat(view_pos.shape[0],1,1)                                                               # [S,N,3]
        out_dirs = (self.cam_pos - view_pos.unsqueeze(1))
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
        shading_buffer = self.split_model_inputs(input=shading_input, total_pixels=in_dirs.shape[1], split_size=1000)
        
        shading_output = []
        for split_buffer in shading_buffer:
            shading_output.append(self.shader(**split_buffer))
        
        masked_rgb_pixels = torch.cat(shading_output, dim=1)

        return masked_rgb_pixels
    
    def importance_sampling(self, hdri_map, num_samples):
        
        # 1. 预处理HDRI，生成CDF
        # 计算亮度（例如，使用RGB通道的加权和）(Rec. 709 or sRGB color space)
        luminance = 0.2126 * hdri_map[:,:,:,0] + 0.7152 * hdri_map[:,:,:,1] + 0.0722 * hdri_map[:,:,:,2]

        # 计算每个像素的权重（亮度乘以正弦因子，以考虑球坐标上的面积变化）
        height, width = luminance.shape[1], luminance.shape[2]
        sin_theta = torch.sin(torch.linspace(0, 0.5 * np.pi, steps=height)).to(hdri_map.device)[:, None]
        weights = luminance * sin_theta

        # 将权重展平并计算CDF
        weights_flat = weights.flatten(start_dim=1)
        cdf = torch.cumsum(weights_flat, dim=1)
        cdf /= cdf.max()  # 归一化到[0, 1]
        
        # 2. 从HDRI光照贴图上采样一个方向
        # 生成均匀分布的随机数
        uniform = torch.rand(cdf.shape[0], num_samples).to(hdri_map.device)
        
        # 使用CDF反向查找找到对应的索引
        idx = torch.searchsorted(cdf, uniform)
        
        # 将一维索引转换为二维索引
        v = idx // width
        u = idx % width

        # 转换为球面坐标（θ，φ）
        theta = (v + 0.5) / height * (0.5 * np.pi)
        phi = (u + 0.5) / width * 2 * np.pi

        # 将球面坐标转换为方向向量
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.cos(theta)
        z = torch.sin(theta) * torch.sin(phi)

        sampled_direction = torch.cat([x[...,None], y[...,None], z[...,None]], dim=-1)

        # 3. 计算给定方向的PDF（概率密度函数）
        # 找到对应的像素索引
        batch_indices = torch.arange(u.shape[0]).reshape(-1, 1).expand_as(u)

        # PDF计算（权重归一化）
        weight = weights[batch_indices, v, u]
        sin_theta = torch.sin(theta)
        pdf = weight / (weights.sum() * 2 * np.pi * np.pi * sin_theta)

        # Sampling the hdri environment map
        sampled_hdri_map = hdri_map[batch_indices, v, u]
        sampled_hdri_map = sampled_hdri_map / pdf[...,None]
        
        return sampled_hdri_map, sampled_direction
    
    def uniform_sampling(self, hdri_map, num_samples):
        
        B, height, width, _ = hdri_map.shape
        
        # 生成均匀分布的phi角（纬度角），范围 [0, pi]
        theta = torch.acos(1 - torch.rand(B, num_samples).to(hdri_map.device))
        
        # 生成均匀分布的theta角（经度角），范围 [0, 2*pi]
        phi = 2 * torch.pi * torch.rand(B, num_samples).to(hdri_map.device)
        
        # 将球面坐标转换为方向向量
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.cos(theta)
        z = torch.sin(theta) * torch.sin(phi)

        sampled_direction = torch.cat([x[...,None], y[...,None], z[...,None]], dim=-1)
        sampled_direction = torch.nn.functional.normalize(sampled_direction, dim=-1)
        
        # 将theta和phi映射到图像的像素坐标
        u = ((phi / (2 * torch.pi)) * width - 0.5).int()
        v = ((theta / torch.pi) * height - 0.5).int()
        
        batch_indices = torch.arange(B).reshape(-1, 1).expand_as(u)
        sampled_hdri_map = hdri_map[batch_indices, v, u]
        
        return sampled_hdri_map, sampled_direction

    def sample_hdri(self, cdf, width, height, num_samples, uniform_sampling=False):
        """
        从HDRI光照贴图上进行重要性采样。
        """
        # 生成均匀分布的随机数
        uniform = torch.rand(cdf.shape[0], num_samples).to(cdf.device)

        if uniform_sampling:
            idx = num_samples
        else:
            # 使用CDF反向查找找到对应的索引
            idx = torch.searchsorted(cdf, uniform)

        # 将一维索引转换为二维索引
        v = idx // width
        u = idx % width

        # 转换为球面坐标（θ，φ）
        theta = (v + 0.5) / height * (0.5 * np.pi)
        phi = (u + 0.5) / width * 2 * np.pi

        # 将球面坐标转换为方向向量
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.cos(theta)
        z = torch.sin(theta) * torch.sin(phi)

        direction = torch.cat([x[...,None], y[...,None], z[...,None]], dim=-1)
        
        return direction

    def compute_pdf(self, weights, width, height, direction):
        """
        计算给定方向的PDF(概率密度函数)。
        """
        # 将方向转换为θ和φ
        theta = torch.acos(direction[...,1])  # y方向的反余弦
        phi = torch.atan2(direction[...,2], direction[...,0]) % (2 * np.pi)

        # 找到对应的像素索引
        u = (phi / (2 * np.pi) * width - 0.5).int()
        v = (theta / (0.5 * np.pi) * height - 0.5).int()
        batch_indices = torch.arange(u.shape[0]).reshape(-1, 1).expand_as(u)

        # PDF计算（权重归一化）
        weight = weights[batch_indices, v, u]
        sin_theta = torch.sin(theta)
        pdf = weight / (weights.sum() * 2 * np.pi * np.pi * sin_theta)

        return pdf[...,None]
    
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
