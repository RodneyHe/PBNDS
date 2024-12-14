import sys, os, math, json
sys.path.append('..')
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

from tqdm import tqdm
from utils.io import load_sdr, load_hdr

import torch
import torch.nn as nn
import torchvision.transforms.functional as tvf

from dataloader.ffhq_pbr import FFHQPBR
from torch.utils.data.dataloader import DataLoader

from models.positional_embedder import get_embedder
from models.neural_shader import FC

# Metrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from models.neural_renderer import NeuralRenderer
from models.unet import UNet128
from models.classical_renderer import GGXShader, BlinnPhongShader

class NDS(nn.Module):
    def __init__(self):
        super().__init__()
        
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
        diffuse_shading_input = self.fourier_feature_transform(position)
        diffue_feature = self.diffuse(diffuse_shading_input)
        color = self.specular(torch.cat([diffue_feature, normal, out_dirs], dim=-1))
        
        return color.clamp(min=0.,max=1.)

class NDR(nn.Module):
    def __init__(self, nds_weights=None):
        super().__init__()
        
        self.shader = NDS()
        if nds_weights is not None:
            self.shader.load_state_dict(nds_weights)
        
        # Optimizer
        self.adam_optimizer = torch.optim.Adam(params=self.shader.parameters(), lr=5e-5)
        
        # Loss function
        self.rec_loss = nn.L1Loss()
    
    def forward(self, position, normal, out_dirs):
        rgb_pred = self.shader(position, normal, out_dirs)
        return rgb_pred
    
    def fit(self, position_gt, normal_gt, rgb_gt, out_dirs_gt, save_path):
        
        pbar = tqdm(range(3000), ncols=80)
        for step in pbar:
            rgb_pred = self.forward(position_gt, normal_gt, out_dirs_gt)
            
            rec_loss = self.rec_loss(rgb_pred, rgb_gt)
            
            if rec_loss.item() is not None:
                self.adam_optimizer.zero_grad()
                rec_loss.backward()
                self.adam_optimizer.step()
        
        torch.save(self.shader.state_dict(), save_path)
        
        return rgb_pred

class PBNDS(nn.Module):
    def __init__(self, pbnds_weights=None):
        super().__init__()
        
        neural_renderer = NeuralRenderer()
        shadow_estimator = UNet128(in_chns=6, out_chns=1)
        
        if pbnds_weights is not None:
            renderer_weights = torch.load(f'{pbnds_weights}/NeuralRenderer.pth')
            shadow_weights = torch.load(f'{pbnds_weights}/ShadowEstimator.pth')
            
            neural_renderer.load_state_dict({k: v for k, v in renderer_weights.items() if 'unet' not in k})
            shadow_estimator.load_state_dict(shadow_weights)
        
        self.neural_renderer = neural_renderer
        self.shadow_estimator = shadow_estimator
    
    def forward(self, render_buffer, num_light_samples, mask):
        
        rgb_pred = self.neural_renderer(render_buffer, num_light_samples)
        
        device = render_buffer['normal_gt'].device
        
        noraml = torch.zeros(128,128,3).to(device)
        noraml[mask] = render_buffer['normal_gt']
        
        rgb_rec = torch.zeros(128,128,3).to(device)
        rgb_rec[mask] = rgb_pred
        
        shadow_pred = self.shadow_estimator(torch.cat([rgb_rec[None], noraml[None]], dim=-1).permute(0,3,1,2))
        
        return rgb_pred, shadow_pred.permute(0,2,3,1)

class Tester:
    def __init__(self, dataset_path):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load shader
        self.ndr = NDR().to(self.device)
        self.pbnds = PBNDS(pbnds_weights='../output/exp_03/weights').to(self.device)
        self.ggx_shader = GGXShader()
        self.blinphong_shader = BlinnPhongShader()
        
        self.width, self.height = 128, 128
        self.dataset_path = dataset_path
        
        cam_pos = torch.tensor([0., 0., 0.])[None, None, :]
        self.cam_pos = nn.Parameter(cam_pos, requires_grad=False).to(self.device)
        
        if not os.path.exists(f'{dataset_path}/nds_weights'):
            os.mkdir(f'{dataset_path}/nds_weights')
        
        dataset = FFHQPBR(data_path=dataset_path, mode='test')
        self.dataset_loader = DataLoader(dataset, batch_size=1, shuffle=False, 
                                         num_workers=0, pin_memory=True)
        
        # Metrics
        self.fid = FrechetInceptionDistance(feature=768, normalize=True, 
                                            input_img_size=(3,128,128),
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
    
    def test(self, model):
        
        dataset_iter = iter(self.dataset_loader)
        
        mse_list = []
        psnr_list = []
        ssim_list = []
        lpips_list = []
        pbar = tqdm(range(len(dataset_iter)), ncols=80)
        for step in pbar:
            data_buffer = next(dataset_iter)
            
            mask_gt = data_buffer['mask_gt'].to(self.device)
            rgb_gt = data_buffer['rgb_gt'].to(self.device)
            albedo_gt = data_buffer['albedo_gt'].to(self.device)
            roughness_gt = data_buffer['roughness_gt'].to(self.device)
            specular_gt = data_buffer['specular_gt'].to(self.device)
            normal_gt = data_buffer['normal_gt'].to(self.device)
            view_pos_gt = data_buffer['view_pos_gt'].to(self.device)
            hdri_gt = data_buffer['hdri_gt'].to(self.device)
            file_index = data_buffer['file_index'][0]
            
            render_buffer = {
                'rgb_gt': rgb_gt[mask_gt],
                'normal_gt': normal_gt[mask_gt],
                'albedo_gt': albedo_gt[mask_gt],
                'roughness_gt': roughness_gt[mask_gt],
                'specular_gt': specular_gt[mask_gt],
                'view_pos_gt': view_pos_gt[mask_gt],
                'hdri_gt': hdri_gt
            }
            
            out_dirs = self.cam_pos - view_pos_gt
            out_dirs = nn.functional.normalize(out_dirs, dim=-1)
            
            # Sampling the HDRi environment map
            sampled_hdri_map, sampled_direction = self.pbnds.neural_renderer.uniform_sampling(hdri_map=hdri_gt, num_samples=128)
            
            if model == 'nds':
                
                view_pos = torch.nn.functional.normalize(view_pos_gt, dim=-1)
                
                if not os.path.exists(f'{self.dataset_path}/nds_weights/{file_index}.pth'):
                    nds_pred_rgb = self.ndr.fit(view_pos[mask_gt], 
                                                normal_gt[mask_gt], 
                                                rgb_gt[mask_gt], 
                                                out_dirs[mask_gt], 
                                                save_path=f'{self.dataset_path}/nds_weights/{file_index}.pth')
                else:
                    nds_weights = torch.load(f'{self.dataset_path}/nds_weights/{file_index}.pth')
                    self.ndr.shader.load_state_dict(nds_weights)
            
                with torch.no_grad():
                    
                    # Test NDS
                    nds_pred_rgb = self.ndr(view_pos[mask_gt], 
                                            normal_gt[mask_gt], 
                                            out_dirs[mask_gt])
                    
                    rgb_vis = torch.zeros(1,128,128,3).to(self.device)
                    rgb_vis[mask_gt] = nds_pred_rgb
                    
                    self._test_FID(pred=rgb_vis.permute(0,3,1,2), gt=rgb_gt.permute(0,3,1,2))
                    mse_list.append(self._test_MSE(pred=nds_pred_rgb, gt=render_buffer['rgb_gt']).item())
                    psnr_list.append(self._test_PSNR(pred=rgb_vis, gt=rgb_gt).item())
                    ssim_list.append(self._test_SSIM(pred=rgb_vis.permute(0,3,1,2), gt=rgb_gt.permute(0,3,1,2)).item())
                    lpips_list.append(self._test_LPIPS(pred=rgb_vis.permute(0,3,1,2), gt=rgb_gt.permute(0,3,1,2)).item())
            
            elif model == 'pbnds':
                
                with torch.no_grad():
                    pbnds_pred_rgb, shadow_pred = self.pbnds(render_buffer=render_buffer, num_light_samples=128, mask=mask_gt[0])
                
                unshadow = torch.zeros(128,128,3).to(device)
                unshadow[mask_gt[0]] = pbnds_pred_rgb
                unshadow = unshadow.permute(2,0,1)
                
                shadow_map = shadow_pred[0].permute(2,0,1)
                
                shadowed = unshadow * shadow_map

                self._test_FID(pred=shadowed[None], gt=rgb_gt.permute(0,3,1,2))
                mse_list.append(self._test_MSE(pred=pbnds_pred_rgb, gt=render_buffer['rgb_gt']).item())
                psnr_list.append(self._test_PSNR(pred=shadowed[None], gt=rgb_gt.permute(0,3,1,2)).item())
                ssim_list.append(self._test_SSIM(pred=shadowed[None], gt=rgb_gt.permute(0,3,1,2)).item())
                lpips_list.append(self._test_LPIPS(pred=shadowed[None], gt=rgb_gt.permute(0,3,1,2)).item())
            
            elif model == 'ggx':
                
                in_dirs = sampled_direction.repeat(render_buffer['view_pos_gt'].shape[0],1,1).to(self.device)
                out_dirs = out_dirs[mask_gt].unsqueeze(1).broadcast_to(in_dirs.shape)
                
                albedo = render_buffer['albedo_gt'].unsqueeze(1).broadcast_to(in_dirs.shape)
                roughness = render_buffer['roughness_gt'].unsqueeze(1)[...,None].broadcast_to(in_dirs.shape)
                specular = render_buffer['specular_gt'].unsqueeze(1)[...,None].broadcast_to(in_dirs.shape)
                normal = render_buffer['normal_gt'].unsqueeze(1).broadcast_to(in_dirs.shape)
                light = sampled_hdri_map.broadcast_to(in_dirs.shape)
                
                ggx_pred_rgb = self.ggx_shader.render_equation(albedo, roughness, specular, normal, out_dirs, in_dirs, light)
                
                rgb_vis = torch.zeros(1,128,128,3).to(self.device)
                rgb_vis[mask_gt] = ggx_pred_rgb
                
                tvf.to_pil_image(rgb_vis[0].permute(2,0,1)).save('./test.png')
                
                self._test_FID(pred=rgb_vis.permute(0,3,1,2), gt=rgb_gt.permute(0,3,1,2))
                mse_list.append(self._test_MSE(pred=ggx_pred_rgb, gt=render_buffer['rgb_gt']).item())
                psnr_list.append(self._test_PSNR(pred=ggx_pred_rgb, gt=render_buffer['rgb_gt']).item())
                ssim_list.append(self._test_SSIM(pred=rgb_vis.permute(0,3,1,2), gt=rgb_gt.permute(0,3,1,2)).item())
                lpips_list.append(self._test_LPIPS(pred=rgb_vis.permute(0,3,1,2), gt=rgb_gt.permute(0,3,1,2)).item())
                
            elif model == 'blinn-phong':
                
                in_dirs = sampled_direction.repeat(render_buffer['view_pos_gt'].shape[0],1,1).to(self.device)
                out_dirs = out_dirs[mask_gt].unsqueeze(1).broadcast_to(in_dirs.shape)
                
                albedo = render_buffer['albedo_gt'].unsqueeze(1).broadcast_to(in_dirs.shape)
                roughness = render_buffer['roughness_gt'].unsqueeze(1)[...,None].broadcast_to(in_dirs.shape)
                specular = render_buffer['specular_gt'].unsqueeze(1)[...,None].broadcast_to(in_dirs.shape)
                normal = render_buffer['normal_gt'].unsqueeze(1).broadcast_to(in_dirs.shape)
                light = sampled_hdri_map.broadcast_to(in_dirs.shape)
                
                with torch.no_grad():
                    blinnphong_pred_rgb = self.blinphong_shader.render_equation(albedo, specular, normal, out_dirs, in_dirs, light, 16)
                
                rgb_vis = torch.zeros(1,128,128,3).to(self.device)
                rgb_vis[mask_gt] = blinnphong_pred_rgb
                
                self._test_FID(pred=rgb_vis.permute(0,3,1,2), gt=rgb_gt.permute(0,3,1,2))
                mse_list.append(self._test_MSE(pred=blinnphong_pred_rgb, gt=render_buffer['rgb_gt']).item())
                psnr_list.append(self._test_PSNR(pred=blinnphong_pred_rgb, gt=render_buffer['rgb_gt']).item())
                ssim_list.append(self._test_SSIM(pred=rgb_vis.permute(0,3,1,2), gt=rgb_gt.permute(0,3,1,2)).item())
                lpips_list.append(self._test_LPIPS(pred=rgb_vis.permute(0,3,1,2), gt=rgb_gt.permute(0,3,1,2)).item())
        
        # Save metric data
        torch.save(torch.tensor(mse_list), f'./eval_result01/mse_{model}.pth')
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

# Helper function
def get_view_pos(depth, width, height, fov):
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

def load_data(data_folder):
    
    data_index = data_folder[-5:]
    
    rgb_gt = load_sdr(f'{data_folder}/{data_index}.png')
    normal_gt = load_sdr(f'{data_folder}/normal_{data_index}.png')
    normal_gt = (normal_gt * 2 - 1.).float()
    albedo_gt = load_sdr(f'{data_folder}/albedo_{data_index}.png')
    roughness_gt = load_sdr(f'{data_folder}/roughness_{data_index}.png')
    specular_gt = load_sdr(f'{data_folder}/specular_{data_index}.png')
    depth_gt = load_hdr(f'{data_folder}/depth_{data_index}.exr')[...,0]
    hdri_gt = load_hdr(f'{data_folder}/hdri_{data_index}.exr', resize=False)
    mask_gt = (rgb_gt != 0)[...,0]
    
    out_dict = {
        'rgb': rgb_gt,
        'normal': normal_gt,
        'albedo': albedo_gt,
        'roughness': roughness_gt,
        'specular': specular_gt,
        'depth': depth_gt,
        'hdri': hdri_gt,
        'mask': mask_gt
    }
    
    return out_dict

def load_all_test_data(data_folder):
    
    subfolder_list = sorted(os.listdir(data_folder))
        
    test_data_list = {}
    for sub_folder in subfolder_list:
        subfolder_path = os.path.join(data_folder, sub_folder)
        test_data_list[sub_folder] = load_data(subfolder_path)
    
    return test_data_list

if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # # Fit and test NDS model
    # ndr = NDR().to(device)
    # ndr.eval()
    
    cam_pos = torch.tensor([0., 0., 0.])[None, None, :].to(device)
    
    # Load all test data
    test_data_dict = load_all_test_data(data_folder='test_data')
    
    # Load predicted fov data
    with open('../dataset/ffhq256_pbr/pred_fov.json', 'r') as openfile:
        pred_fov_dict = json.load(openfile)
    
    # for test_data_index, test_data in test_data_dict.items():
        
    #     if os.path.exists(f'result/{test_data_index}'):
    #         continue
    #     else:
    #         os.mkdir(f'result/{test_data_index}')
        
    #     print(f'Fitting NDR model for data {test_data_index}.')
        
    #     pred_fov = pred_fov_dict[test_data_index]
        
    #     view_pos = get_view_pos(depth=test_data['depth'], width=128, height=128, fov=pred_fov).to(device)
        
    #     out_dirs = (cam_pos - view_pos).to(device)
    #     out_dirs = nn.functional.normalize(out_dirs, dim=-1)
        
    #     normal = test_data['normal'].to(device)
    #     rgb = test_data['rgb'].to(device)
    #     mask = test_data['mask'].to(device)
        
    #     view_pos = torch.nn.functional.normalize(view_pos, dim=-1)
        
    #     rgb_pred = ndr.fit(position_gt=view_pos[mask], normal_gt=normal[mask], rgb_gt=rgb[mask], out_dirs_gt=out_dirs[mask],
    #                        save_path=f'test_data/{test_data_index}/nds_{test_data_index}.pth')
        
    #     rgb_vis = torch.zeros(128,128,3).to(device)
    #     rgb_vis[mask] = rgb_pred
    #     rgb_vis = rgb_vis.detach().cpu().permute(2,0,1)
        
    #     tvf.to_pil_image(rgb_vis).save(f'result/{test_data_index}/nds_pred_{test_data_index}.png')
    
    # Test PBNDS model
    # pbnds = PBNDS(pbnds_weights='../output/exp_03/weights').to(device)
    # pbnds.eval()
    
    # for test_data_index, test_data in test_data_dict.items():
        
    #     print(f'Test PBNDR model for data {test_data_index}.')
        
    #     pred_fov = pred_fov_dict[test_data_index]
    #     view_pos = get_view_pos(depth=test_data['depth'], width=128, height=128, fov=pred_fov).to(device)
    #     mask_gt = test_data['mask']
        
    #     render_buffer = {
    #         'rgb_gt': test_data['rgb'][mask_gt].to(device),
    #         'normal_gt': test_data['normal'][mask_gt].to(device),
    #         'albedo_gt': test_data['albedo'][mask_gt].to(device),
    #         'roughness_gt': test_data['roughness'][mask_gt].to(device),
    #         'specular_gt': test_data['specular'][mask_gt].to(device),
    #         'view_pos_gt': view_pos[mask_gt].to(device),
    #         'hdri_gt': test_data['hdri'][None].to(device)
    #     }
        
    #     with torch.no_grad():
    #         rgb_pred, shadow_pred = pbnds(render_buffer=render_buffer, num_light_samples=128, mask=mask_gt)
        
    #     unshadow = torch.zeros(128,128,3).to(device)
    #     unshadow[mask_gt] = rgb_pred
    #     unshadow = unshadow.detach().cpu().permute(2,0,1)
        
    #     shadow_map = shadow_pred[0].detach().cpu().permute(2,0,1)
        
    #     shadowed = unshadow * shadow_map
        
    #     tvf.to_pil_image(unshadow).save(f'result/{test_data_index}/pbnds_unshadow_pred_{test_data_index}.png')
    #     tvf.to_pil_image(shadow_map).save(f'result/{test_data_index}/pbnds_shadow_pred_{test_data_index}.png')
    #     tvf.to_pil_image(shadowed).save(f'result/{test_data_index}/pbnds_shadowed_pred_{test_data_index}.png')
    #     del pred_fov, view_pos, mask_gt, render_buffer, unshadow,shadow_map, shadowed
    #     torch.cuda.empty_cache()
    
    # Test GGX renderer
    # ggx_shader = GGXShader()
    # cam_pos = torch.tensor([0., 0., 0.])[None, None, :].to(device)
    
    # for test_data_index, test_data in test_data_dict.items():
        
    #     print(f'Test GGX model for data {test_data_index}.')
        
    #     mask_gt = test_data['mask']
    #     pred_fov = pred_fov_dict[test_data_index]
    #     view_pos = get_view_pos(depth=test_data['depth'], width=128, height=128, fov=pred_fov).to(device)
    #     view_pos = view_pos[mask_gt]
        
    #     # Sampling the HDRi environment map
    #     sampled_hdri_map, sampled_direction = pbnds.neural_renderer.uniform_sampling(hdri_map=test_data['hdri'][None], num_samples=128)
        
    #     in_dirs = sampled_direction.repeat(view_pos.shape[0],1,1).to(device)
    #     out_dirs = (cam_pos - view_pos.unsqueeze(1))
    #     out_dirs = nn.functional.normalize(out_dirs, dim=-1)
    #     out_dirs = out_dirs.broadcast_to(in_dirs.shape)
        
    #     albedo = test_data['albedo'][mask_gt].to(device)
    #     albedo = albedo.unsqueeze(1).broadcast_to(in_dirs.shape)
    #     roughness = test_data['roughness'][mask_gt].to(device)
    #     roughness = roughness.unsqueeze(1)[...,None].broadcast_to(in_dirs.shape)
    #     specular = test_data['specular'][mask_gt].to(device)
    #     specular = specular.unsqueeze(1)[...,None].broadcast_to(in_dirs.shape)
    #     normal = test_data['normal'][mask_gt].to(device)
    #     normal = normal.unsqueeze(1).broadcast_to(in_dirs.shape)
        
    #     light = sampled_hdri_map.broadcast_to(in_dirs.shape).to(device)
        
    #     rgb_render = ggx_shader.render_equation(albedo, roughness, specular, normal, out_dirs, in_dirs, light)
        
    #     rgb_vis = torch.zeros(128,128,3).to(device)
    #     rgb_vis[mask_gt] = rgb_render
    #     rgb_vis = rgb_vis.detach().cpu().permute(2,0,1)
        
    #     tvf.to_pil_image(rgb_vis).save(f'result/{test_data_index}/ggx_pred_{test_data_index}.png')
    
    # Test Blin-Phong renderer
    # blinphong_shader = BlinnPhongShader()
    # cam_pos = torch.tensor([0., 0., 0.])[None, None, :].to(device)
    
    # for test_data_index, test_data in test_data_dict.items():
        
    #     print(f'Test  model for data {test_data_index}.')
        
    #     mask_gt = test_data['mask']
    #     pred_fov = pred_fov_dict[test_data_index]
    #     view_pos = get_view_pos(depth=test_data['depth'], width=128, height=128, fov=pred_fov).to(device)
    #     view_pos = view_pos[mask_gt]
        
    #     # Sampling the HDRi environment map
    #     sampled_hdri_map, sampled_direction = pbnds.neural_renderer.uniform_sampling(hdri_map=test_data['hdri'][None], num_samples=128)
        
    #     in_dirs = sampled_direction.repeat(view_pos.shape[0],1,1).to(device)
    #     out_dirs = (cam_pos - view_pos.unsqueeze(1))
    #     out_dirs = nn.functional.normalize(out_dirs, dim=-1)
    #     out_dirs = out_dirs.broadcast_to(in_dirs.shape)
        
    #     albedo = test_data['albedo'][mask_gt].to(device)
    #     albedo = albedo.unsqueeze(1).broadcast_to(in_dirs.shape)
    #     roughness = test_data['roughness'][mask_gt].to(device)
    #     roughness = roughness.unsqueeze(1)[...,None].broadcast_to(in_dirs.shape)
    #     specular = test_data['specular'][mask_gt].to(device)
    #     specular = specular.unsqueeze(1)[...,None].broadcast_to(in_dirs.shape)
    #     normal = test_data['normal'][mask_gt].to(device)
    #     normal = normal.unsqueeze(1).broadcast_to(in_dirs.shape)
        
    #     light = sampled_hdri_map.broadcast_to(in_dirs.shape).to(device)
        
    #     rgb_render = blinphong_shader.render_equation(albedo, specular, normal, out_dirs, in_dirs, light, m=16)
        
    #     rgb_vis = torch.zeros(128,128,3).to(device)
    #     rgb_vis[mask_gt] = rgb_render
    #     rgb_vis = rgb_vis.detach().cpu().permute(2,0,1)
        
    #     tvf.to_pil_image(rgb_vis).save(f'result/{test_data_index}/BlinPhong_pred_{test_data_index}.png')
    
    tester = Tester(dataset_path='../dataset/ffhq256_pbr')
    tester.test('nds')
    tester.test('pbnds')
    tester.test('ggx')
    tester.test('blinn-phong')
