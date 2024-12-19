import sys, os, math, json
sys.path.append('..')
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

from tqdm import tqdm
from utils.io import load_sdr, load_hdr
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
        
        self.pbndr = PBNDR(pretrained_weights='pretrained_weights/pbndr_weights').to(self.device)
        
        self.ggx_shader = GGXShader()
        
        self.blinphong_shader = BlinnPhongShader()
        
        if not os.path.exists(f'pretrained_weights/ndr_weights'):
            os.mkdir(f'pretrained_weights/ndr_weights')
        
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
            
            # Calculate outbound direction
            out_dirs_gt = self.cam_pos - pos_in_cam_gt.unsqueeze(1)
            out_dirs_gt = nn.functional.normalize(out_dirs_gt, dim=-1)
            
            if model == 'nds':
                
                nds_weights_path = f'pretrained_weights/ndr_weights/{file_index}.pth'
                if not os.path.exists(nds_weights_path):
                    nds_pred_rgb = self.ndr.fit(position=pos_in_cam_gt, 
                                                normal=normal_gt,
                                                out_dirs=out_dirs_gt.squeeze(1),
                                                save_path=nds_weights_path)
                else:
                    nds_weights = torch.load(nds_weights_path)
                    self.ndr.load_state_dict(nds_weights)
            
                with torch.no_grad():
                    
                    # Test NDS
                    nds_pred_rgb = self.ndr(position=pos_in_cam_gt, 
                                            normal=normal_gt,
                                            out_dirs=out_dirs_gt.squeeze(1))
                    
                    rgb_vis = torch.zeros(1,128,128,3).to(self.device)
                    rgb_vis[mask_gt] = nds_pred_rgb
                    
                    self._test_FID(pred=rgb_vis.permute(0,3,1,2), gt=rgb_gt.permute(0,3,1,2))
                    mse_list.append(self._test_MSE(pred=nds_pred_rgb, gt=rgb_gt).item())
                    psnr_list.append(self._test_PSNR(pred=rgb_vis, gt=rgb_gt).item())
                    ssim_list.append(self._test_SSIM(pred=rgb_vis.permute(0,3,1,2), gt=rgb_gt.permute(0,3,1,2)).item())
                    lpips_list.append(self._test_LPIPS(pred=rgb_vis.permute(0,3,1,2), gt=rgb_gt.permute(0,3,1,2)).item())
            
            if model == 'pbnds':
                
                render_buffer = {
                    'normal_gt': normal_gt[mask_gt],
                    'albedo_gt': albedo_gt[mask_gt],
                    'roughness_gt': roughness_gt[mask_gt],
                    'specular_gt': specular_gt[mask_gt],
                    'pos_in_cam_gt': pos_in_cam_gt[mask_gt],
                    'hdri_gt': hdri_gt
                }
                
                with torch.no_grad():
                    pbnds_pred_rgb, shadow_pred = self.pbndr(render_buffer=render_buffer, 
                                                             mask=mask_gt[0],
                                                             num_light_samples=128)
                
                unshadow = torch.zeros(128,128,3).to(self.device)
                unshadow[mask_gt[0]] = pbnds_pred_rgb
                unshadow = unshadow.permute(2,0,1)
                
                shadow_map = shadow_pred[0].permute(2,0,1)
                
                shadowed = unshadow * shadow_map

                self._test_FID(pred=shadowed[None], gt=rgb_gt.permute(0,3,1,2))
                mse_list.append(self._test_MSE(pred=shadowed[None], gt=rgb_gt.permute(0,3,1,2)).item())
                psnr_list.append(self._test_PSNR(pred=shadowed[None], gt=rgb_gt.permute(0,3,1,2)).item())
                ssim_list.append(self._test_SSIM(pred=shadowed[None], gt=rgb_gt.permute(0,3,1,2)).item())
                lpips_list.append(self._test_LPIPS(pred=shadowed[None], gt=rgb_gt.permute(0,3,1,2)).item())
                
                rgb_vis = shadowed
            
            if model == 'ggx':
                
                # Sampling the HDRi environment map
                sampled_hdri_map, sampled_direction = self.sampler.uniform_sampling(hdri_map=hdri_gt, num_samples=128)
                
                in_dirs_gt = sampled_direction.repeat(pos_in_cam_gt.shape[0],1,1)
                
                shading_input = {
                    'normal': normal_gt[mask_gt].unsqueeze(1).broadcast_to(in_dirs_gt.shape),
                    'albedo': albedo_gt[mask_gt].unsqueeze(1).broadcast_to(in_dirs_gt.shape),
                    'roughness': roughness_gt[mask_gt].unsqueeze(1)[...,None].broadcast_to(*in_dirs_gt.shape[:-1],1),
                    'specular': specular_gt[mask_gt].unsqueeze(1)[...,None].broadcast_to(*in_dirs_gt.shape[:-1],1),
                    'in_dirs': in_dirs_gt,
                    'out_dirs': out_dirs_gt[mask_gt].broadcast_to(in_dirs_gt.shape),
                    'hdri_samples': sampled_hdri_map.broadcast_to(in_dirs_gt.shape)
                }
                
                ggx_pred_rgb = self.ggx_shader.render_equation(shading_input)
                
                rgb_vis = torch.zeros(1,128,128,3).to(self.device)
                rgb_vis[mask_gt] = ggx_pred_rgb
                
                tvf.to_pil_image(rgb_vis[0].permute(2,0,1)).save('./test.png')
                
                self._test_FID(pred=rgb_vis.permute(0,3,1,2), gt=rgb_gt.permute(0,3,1,2))
                mse_list.append(self._test_MSE(pred=rgb_vis, gt=rgb_gt).item())
                psnr_list.append(self._test_PSNR(pred=rgb_vis, gt=rgb_gt).item())
                ssim_list.append(self._test_SSIM(pred=rgb_vis.permute(0,3,1,2), gt=rgb_gt.permute(0,3,1,2)).item())
                lpips_list.append(self._test_LPIPS(pred=rgb_vis.permute(0,3,1,2), gt=rgb_gt.permute(0,3,1,2)).item())
                
            if model == 'blinn-phong':
                
                # Sampling the HDRi environment map
                sampled_hdri_map, sampled_direction = self.sampler.uniform_sampling(hdri_map=hdri_gt, num_samples=128)
                
                in_dirs_gt = sampled_direction.repeat(pos_in_cam_gt.shape[0],1,1)
                
                shading_input = {
                    'normal': normal_gt[mask_gt].unsqueeze(1).broadcast_to(in_dirs_gt.shape),
                    'albedo': albedo_gt[mask_gt].unsqueeze(1).broadcast_to(in_dirs_gt.shape),
                    'roughness': roughness_gt[mask_gt].unsqueeze(1)[...,None].broadcast_to(*in_dirs_gt.shape[:-1],1),
                    'specular': specular_gt[mask_gt].unsqueeze(1)[...,None].broadcast_to(*in_dirs_gt.shape[:-1],1),
                    'in_dirs': in_dirs_gt,
                    'out_dirs': out_dirs_gt[mask_gt].broadcast_to(in_dirs_gt.shape),
                    'hdri_samples': sampled_hdri_map.broadcast_to(in_dirs_gt.shape)
                }
                
                with torch.no_grad():
                    blinnphong_pred_rgb = self.blinphong_shader.render_equation(shading_input, 16)
                
                rgb_vis = torch.zeros(1,128,128,3).to(self.device)
                rgb_vis[mask_gt] = blinnphong_pred_rgb
                
                self._test_FID(pred=rgb_vis.permute(0,3,1,2), gt=rgb_gt.permute(0,3,1,2))
                mse_list.append(self._test_MSE(pred=rgb_vis, gt=rgb_gt).item())
                psnr_list.append(self._test_PSNR(pred=rgb_vis, gt=rgb_gt).item())
                ssim_list.append(self._test_SSIM(pred=rgb_vis.permute(0,3,1,2), gt=rgb_gt.permute(0,3,1,2)).item())
                lpips_list.append(self._test_LPIPS(pred=rgb_vis.permute(0,3,1,2), gt=rgb_gt.permute(0,3,1,2)).item())
            
            tvf.to_pil_image(rgb_vis).save(f'{self.result_path}/{model}_{file_index}.png')
        
        if self.save_matrics:
            # Save metric data
            torch.save(torch.tensor(mse_list), f'{self.result_path}/eval_result01/mse_{model}.pth')
            torch.save(torch.tensor(psnr_list), f'./eval_result01/psnr_{model}.pth')
            torch.save(torch.tensor(ssim_list), f'./eval_result01/ssim_{model}.pth')
            torch.save(torch.tensor(lpips_list), f'./eval_result01/lpips_{model}.pth')
        
            report = {
                'mse': sum(mse_list)/ len(mse_list),
                'psnr': sum(psnr_list) / len(psnr_list),
                'ssim': sum(ssim_list) / len(ssim_list),
                'lpips': sum(lpips_list) / len(lpips_list),
                'fid': self.fid.compute().item()
            }
        
            json_object = json.dumps(report, indent=4)
            with open(f'./eval_result01/report_{model}.json', 'w') as outfile:
                outfile.write(json_object)

if __name__ == '__main__':
    
    # dataset = FFHQPBR(data_path=dataset_path, mode='test')
    # sffhqpbr_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    testdata_loader = get_dataloader(data_folder='test_data02')
    
    tester = Tester(data_loader=testdata_loader, configs=None)
    #tester.eval_metrics('nds')
    tester.eval_metrics('pbnds')
    # tester.eval_metrics('ggx')
    # tester.eval_metrics('blinn-phong')
