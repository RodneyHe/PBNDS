import torch
import torch.nn as nn

class GGXShader(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, input_buffer):
        
        render_output = self.render_equation(**input_buffer)
        
        return render_output
    
    def v_schlick_ggx(self, roughness, cos):
        '''
        Geometry term V, V = G / (4 * cos * cos), schlick ggx
        '''
        r2 = ((1 + roughness) ** 2) / 8
        return 0.5 / (cos * (1 - r2) + r2).clamp(min=1e-2)

    def d_sg(self, roughness, cos):
        r2 = (roughness * roughness).clamp(min=1e-2)
        amp = 1 / (r2 * torch.pi)
        sharp = 2 / r2
        return amp * torch.exp(sharp * (cos - 1))
    
    def render_equation(self, shading_input):
        
        albedo = shading_input['albedo']
        roughness = shading_input['roughness']
        specular = shading_input['specular']
        normal = shading_input['normal']
        out_dirs = shading_input['out_dirs']
        in_dirs = shading_input['in_dirs']
        light = shading_input['hdri_samples']
        
        # Diffuse BRDF
        diffuse_brdf = (1 - specular) * albedo / torch.pi
        
        # Diffuse BRDF
        half_dirs = in_dirs + out_dirs
        half_dirs = nn.functional.normalize(half_dirs, dim=-1)
        h_d_n = (half_dirs * normal).sum(dim=-1, keepdim=True).clamp(min=0)
        h_d_o = (half_dirs * out_dirs).sum(dim=-1, keepdim=True).clamp(min=0)
        n_d_i = (normal * in_dirs).sum(dim=-1, keepdim=True).clamp(min=0)
        n_d_o = (normal * out_dirs).sum(dim=-1, keepdim=True).clamp(min=0)
        
        # Fresnel term F (Schlick Approximation)
        F0 = 0.04 * (1 - specular) + albedo * specular
        F = F0 + (1. - F0) * ((1. - h_d_o) ** 5)
        
        # Geometry term with Smiths Approximation
        V = self.v_schlick_ggx(roughness, n_d_i) * self.v_schlick_ggx(roughness, n_d_o)
        
        # Normal distributed function (SG)
        D = self.d_sg(roughness, h_d_n).clamp(max=1)
        
        specular_brdf = D * F * V 
        
        # RGB color shading
        incident_area = torch.ones_like(light) * 2 * torch.pi
        render_output = ((diffuse_brdf + specular_brdf) * light * incident_area * n_d_i).mean(dim=1)
        
        return render_output.clamp(0.,1.)

class BlinnPhongShader(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, input_buffer):
        
        render_output = self.render_equation(**input_buffer)
        
        return render_output
    
    def render_equation(self, shading_input, m):
        
        albedo = shading_input['albedo']
        specular = shading_input['specular']
        normal = shading_input['normal']
        out_dirs = shading_input['out_dirs']
        in_dirs = shading_input['in_dirs']
        light = shading_input['hdri_samples']
        
        # Diffuse BRDF
        half_dirs = in_dirs + out_dirs
        half_dirs = nn.functional.normalize(half_dirs, dim=-1)
        h_d_n = (half_dirs * normal).sum(dim=-1, keepdim=True).clamp(min=0)
        n_d_i = (normal * in_dirs).sum(dim=-1, keepdim=True).clamp(min=0)
        
        brdf = albedo * n_d_i + specular * torch.pow(h_d_n, m) 
        
        # RGB color shading
        incident_area = torch.ones_like(light) * 2 * torch.pi
        render_output = (brdf * light * incident_area * n_d_i).mean(dim=1)
        
        return render_output.clamp(0.,1.)
