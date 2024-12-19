import torch
import numpy as np

class Sampler:
    def __init__(self):
        pass

    def uniform_sampling(self, hdri_map, num_samples):
            
        B, height, width, _ = hdri_map.shape
        
        # Uniformly sampling theta (elevation angle), range [0, 2*pi]
        phi = 2 * torch.pi * torch.rand(B, num_samples).to(hdri_map.device)
        
        # Uniformly sample phi (azimuth angle), range [0, pi]
        theta = torch.acos(2 * torch.rand(B, num_samples) - 1).to(hdri_map.device)
        
        # Convert spherical coordinates to Cartesian coordinates 
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.cos(theta)
        z = torch.sin(theta) * torch.sin(phi)

        # Form the direction vector
        sampled_direction = torch.cat([x[...,None], y[...,None], z[...,None]], dim=-1)
        sampled_direction = torch.nn.functional.normalize(sampled_direction, dim=-1)
        
        # Mapping (phi, theta) to pixel coordinates of an equirectangular environment map
        u = ((phi / (2 * torch.pi)) * (width-1)).long()
        v = ((theta / torch.pi) * (height-1)).long()
        
        batch_indices = torch.arange(B).reshape(-1, 1).expand_as(u)
        sampled_hdri_map = hdri_map[batch_indices, v, u]
        
        return sampled_hdri_map, sampled_direction
    
    def importance_sampling(self, hdri_map, num_samples):
        
        # 1. Preprocess HDRI, generating CDF
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
    