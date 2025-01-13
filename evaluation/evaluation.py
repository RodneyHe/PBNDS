import sys, os, math, json
sys.path.append('..')
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

from tqdm import tqdm
from utils.sampler import Sampler

import torch
import torch.nn as nn
import torchvision.transforms.functional as tvf

from dataloader.ffhq_pbr import FFHQPBR
from torch.utils.data.dataloader import DataLoader
from test_dataloader import get_dataloader

# Metrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

# Models
from models.ndr import NDR
from models.pbndr import PBNDR
from models.classical_renderer import GGXShader, BlinnPhongShader

class Tester:
    def __init__(self, data_loader, configs, save_matrics=False):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configs = configs
        self.save_matrics = save_matrics
        self.result_path = 'result'
        
        self.data_loader = data_loader
        
        # Load sampler
        self.sampler = Sampler()
        
        # Load shader
        self.ndr = NDR().to(self.device)
        self.pbndr = PBNDR().to(self.device)
        self.ggx_shader = GGXShader()
        self.blinphong_shader = BlinnPhongShader()
        
        # Camera position
        cam_pos = torch.tensor([0., 0., 0.])[None, None, :]
        self.cam_pos = nn.Parameter(cam_pos, requires_grad=False).to(self.device)
        
        # Metrics
        self.fid = FrechetInceptionDistance(feature=768, normalize=True, input_img_size=(3,128,128),
                                            compute_on_cpu=True).to(self.device)
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(self.device)
    
    def _test_PSNR(self, pred, gt):
        return self.psnr(pred, gt)
    
    def _test_SSIM(self, pred, gt):
        return self.ssim(pred, gt)
    
    def _test_LPIPS(self, pred, gt):
        return self.lpips(pred, gt)
    
    def _test_MSE(self, pred, gt):
        return torch.nn.functional.mse_loss(pred, gt)
    
    def _test_FID(self, pred, gt):
        self.fid.update(pred, real=False)
        self.fid.update(gt, real=True)
    
    def eval_metrics(self, model):
        
        if not os.path.exists(f'{self.result_path}/{model}'):
            os.mkdir(f'{self.result_path}/{model}')
        
        if not os.path.exists(f'{self.result_path}/{model}/pred_rgb'):
            os.mkdir(f'{self.result_path}/{model}/pred_rgb')
        
        data_iter = iter(self.data_loader)
        
        mse_list = []
        psnr_list = []
        ssim_list = []
        lpips_list = []
        pbar = tqdm(range(len(data_iter)), ncols=80)
        for _ in pbar:
            
            data_buffer = next(data_iter)
            
            mask_gt = data_buffer['mask_gt'].to(self.device)
            rgb_gt = data_buffer['rgb_gt'].to(self.device)
            albedo_gt = data_buffer['albedo_gt'].to(self.device)
            roughness_gt = data_buffer['roughness_gt'].to(self.device)
            specular_gt = data_buffer['specular_gt'].to(self.device)
            normal_gt = data_buffer['normal_gt'].to(self.device)
            pos_in_cam_gt = data_buffer['pos_in_cam_gt'].to(self.device)
            hdri_gt = data_buffer['hdri_gt'].to(self.device)
            file_index = data_buffer['file_index'][0]
            
            if model == 'nds':
                
                # Calculate outbound direction
                out_dirs_gt = self.cam_pos - pos_in_cam_gt.unsqueeze(1)
                out_dirs_gt = nn.functional.normalize(out_dirs_gt, dim=-1)
                
                nds_weights_path = f'../dataset/ffhq256_pbr/nds_weights/{file_index}.pth'
                if not os.path.exists(nds_weights_path):
                    nds_pred_rgb = self.ndr.fit(position=pos_in_cam_gt, 
                                                normal=normal_gt,
                                                out_dirs=out_dirs_gt.squeeze(1),
                                                rgb_gt=rgb_gt,
                                                save_path=nds_weights_path)
                else:
                    nds_weights = torch.load(nds_weights_path)
                    self.ndr.shader.load_state_dict(nds_weights)
            
                with torch.no_grad():
                    
                    # Test NDS
                    nds_pred_rgb = self.ndr(position=pos_in_cam_gt, 
                                            normal=normal_gt,
                                            out_dirs=out_dirs_gt.squeeze(1))
                    
                    rgb_vis = nds_pred_rgb[0].permute(2,0,1)
                    
                    self._test_FID(pred=rgb_vis[None], gt=rgb_gt.permute(0,3,1,2))
                    mse_list.append(self._test_MSE(pred=rgb_vis[None], gt=rgb_gt.permute(0,3,1,2)).item())
                    psnr_list.append(self._test_PSNR(pred=rgb_vis[None], gt=rgb_gt.permute(0,3,1,2)).item())
                    ssim_list.append(self._test_SSIM(pred=rgb_vis[None], gt=rgb_gt.permute(0,3,1,2)).item())
                    lpips_list.append(self._test_LPIPS(pred=rgb_vis[None], gt=rgb_gt.permute(0,3,1,2)).item())
            
            if model == 'pbnds':
                
                render_buffer = {
                    'normal_gt': normal_gt[mask_gt],
                    'albedo_gt': albedo_gt[mask_gt],
                    'roughness_gt': roughness_gt[mask_gt],
                    'specular_gt': specular_gt[mask_gt],
                    'pos_in_cam_gt': pos_in_cam_gt[mask_gt],
                    'hdri_gt': hdri_gt
                }
                
                pbnds_weights_path = 'pretrained_weights/pbndr_weights/NeuralRenderer.pth'
                renderer_weights = torch.load(pbnds_weights_path)
                self.pbndr.neural_render.load_state_dict({k: v for k, v in renderer_weights.items() if 'unet' not in k})
                
                shadow_weights_path = 'pretrained_weights/pbndr_weights/ShadowEstimator.pth'
                if os.path.exists(shadow_weights_path):
                    shadow_weights = torch.load(shadow_weights_path)
                    self.pbndr.shadow_estimator.load_state_dict(shadow_weights)
                    
                    with torch.no_grad():
                        pbnds_pred_rgb, shadow_pred = self.pbndr(render_buffer=render_buffer, mask=mask_gt[0], num_light_samples=128)
                    
                    unshadow = torch.zeros(128,128,3).to(self.device)
                    unshadow[mask_gt[0]] = pbnds_pred_rgb
                    unshadow = unshadow.permute(2,0,1)
                    
                    shadow_map = shadow_pred[0].permute(2,0,1)
                    
                    shadowed = unshadow * shadow_map
                    
                    rgb_vis = torch.cat([shadowed, shadow_map.repeat(3,1,1)], dim=2)
                    
                    pred = shadowed[None]
                
                else:
                    with torch.no_grad():
                        pbnds_pred_rgb = self.pbndr(render_buffer=render_buffer, mask=mask_gt[0], num_light_samples=128, 
                                                    shadowing=False)

                    rgb_vis = torch.zeros(128,128,3).to(self.device)
                    rgb_vis[mask_gt[0]] = pbnds_pred_rgb
                    rgb_vis = rgb_vis.permute(2,0,1)
                    
                    pred = rgb_vis[None]

                self._test_FID(pred=pred, gt=rgb_gt.permute(0,3,1,2))
                mse_list.append(self._test_MSE(pred=pred, gt=rgb_gt.permute(0,3,1,2)).item())
                psnr_list.append(self._test_PSNR(pred=pred, gt=rgb_gt.permute(0,3,1,2)).item())
                ssim_list.append(self._test_SSIM(pred=pred, gt=rgb_gt.permute(0,3,1,2)).item())
                lpips_list.append(self._test_LPIPS(pred=pred, gt=rgb_gt.permute(0,3,1,2)).item())
            
            if model == 'ggx':
                
                # Calculate outbound direction
                out_dirs_gt = self.cam_pos - pos_in_cam_gt[mask_gt].unsqueeze(1)
                out_dirs_gt = nn.functional.normalize(out_dirs_gt, dim=-1)
                
                # Sampling the HDRi environment map
                sampled_hdri_map, sampled_direction = self.sampler.uniform_sampling(hdri_map=hdri_gt, num_samples=128)
                
                in_dirs_gt = sampled_direction.repeat(pos_in_cam_gt[mask_gt].shape[0],1,1)
                
                shading_input = {
                    'normal': normal_gt[mask_gt].unsqueeze(1).broadcast_to(in_dirs_gt.shape),
                    'albedo': albedo_gt[mask_gt].unsqueeze(1).broadcast_to(in_dirs_gt.shape),
                    'roughness': roughness_gt[mask_gt].unsqueeze(1)[...,None].broadcast_to(*in_dirs_gt.shape[:-1],1),
                    'specular': specular_gt[mask_gt].unsqueeze(1)[...,None].broadcast_to(*in_dirs_gt.shape[:-1],1),
                    'in_dirs': in_dirs_gt,
                    'out_dirs': out_dirs_gt.broadcast_to(in_dirs_gt.shape),
                    'hdri_samples': sampled_hdri_map.broadcast_to(in_dirs_gt.shape)
                }
                
                with torch.no_grad():
                    ggx_pred_rgb = self.ggx_shader.render_equation(shading_input)
                
                rgb_vis = torch.zeros(128,128,3).to(self.device)
                rgb_vis[mask_gt[0]] = ggx_pred_rgb
                rgb_vis = rgb_vis.permute(2,0,1)
                
                pred = rgb_vis[None]
                
                self._test_FID(pred=pred, gt=rgb_gt.permute(0,3,1,2))
                mse_list.append(self._test_MSE(pred=pred, gt=rgb_gt.permute(0,3,1,2)).item())
                psnr_list.append(self._test_PSNR(pred=pred, gt=rgb_gt.permute(0,3,1,2)).item())
                ssim_list.append(self._test_SSIM(pred=pred, gt=rgb_gt.permute(0,3,1,2)).item())
                lpips_list.append(self._test_LPIPS(pred=pred, gt=rgb_gt.permute(0,3,1,2)).item())
                
            if model == 'blinn-phong':
                
                # Calculate outbound direction
                out_dirs_gt = self.cam_pos - pos_in_cam_gt[mask_gt].unsqueeze(1)
                out_dirs_gt = nn.functional.normalize(out_dirs_gt, dim=-1)
                
                # Sampling the HDRi environment map
                sampled_hdri_map, sampled_direction = self.sampler.uniform_sampling(hdri_map=hdri_gt, num_samples=128)
                
                in_dirs_gt = sampled_direction.repeat(pos_in_cam_gt[mask_gt].shape[0],1,1)
                
                shading_input = {
                    'normal': normal_gt[mask_gt].unsqueeze(1).broadcast_to(in_dirs_gt.shape),
                    'albedo': albedo_gt[mask_gt].unsqueeze(1).broadcast_to(in_dirs_gt.shape),
                    'roughness': roughness_gt[mask_gt].unsqueeze(1)[...,None].broadcast_to(*in_dirs_gt.shape[:-1],1),
                    'specular': specular_gt[mask_gt].unsqueeze(1)[...,None].broadcast_to(*in_dirs_gt.shape[:-1],1),
                    'in_dirs': in_dirs_gt,
                    'out_dirs': out_dirs_gt.broadcast_to(in_dirs_gt.shape),
                    'hdri_samples': sampled_hdri_map.broadcast_to(in_dirs_gt.shape)
                }
                
                with torch.no_grad():
                    blinnphong_pred_rgb = self.blinphong_shader.render_equation(shading_input, 16)
                
                rgb_vis = torch.zeros(128,128,3).to(self.device)
                rgb_vis[mask_gt[0]] = blinnphong_pred_rgb
                rgb_vis = rgb_vis.permute(2,0,1)
                
                pred = rgb_vis[None]
                
                self._test_FID(pred=pred, gt=rgb_gt.permute(0,3,1,2))
                mse_list.append(self._test_MSE(pred=pred, gt=rgb_gt.permute(0,3,1,2)).item())
                psnr_list.append(self._test_PSNR(pred=pred, gt=rgb_gt.permute(0,3,1,2)).item())
                ssim_list.append(self._test_SSIM(pred=pred, gt=rgb_gt.permute(0,3,1,2)).item())
                lpips_list.append(self._test_LPIPS(pred=pred, gt=rgb_gt.permute(0,3,1,2)).item())
            
            tvf.to_pil_image(rgb_vis).save(f'{self.result_path}/{model}/pred_rgb/pred_{file_index}.png')
        
        if self.save_matrics:
            # Save metric data
            torch.save(torch.tensor(mse_list), f'{self.result_path}/{model}/mse.pth')
            torch.save(torch.tensor(psnr_list), f'{self.result_path}/{model}/psnr.pth')
            torch.save(torch.tensor(ssim_list), f'{self.result_path}/{model}/ssim.pth')
            torch.save(torch.tensor(lpips_list), f'{self.result_path}/{model}/lpips.pth')
        
            report = {
                'mse': sum(mse_list)/ len(mse_list),
                'psnr': sum(psnr_list) / len(psnr_list),
                'ssim': sum(ssim_list) / len(ssim_list),
                'lpips': sum(lpips_list) / len(lpips_list),
                'fid': self.fid.compute().item()
            }
        
            json_object = json.dumps(report, indent=4)
            with open(f'{self.result_path}/{model}/report.json', 'w') as outfile:
                outfile.write(json_object)

if __name__ == '__main__':
    
    dataset = FFHQPBR(data_path='../dataset/ffhq256_pbr', mode='test')
    ffhqpbr_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    #testdata_loader = get_dataloader(data_folder='test_data02')
    
    tester = Tester(data_loader=ffhqpbr_loader, configs=None, save_matrics=True)
    #tester.eval_metrics('nds')
    tester.eval_metrics('pbnds')
    #tester.eval_metrics('ggx')
    #tester.eval_metrics('blinn-phong')
