from tqdm import tqdm
import math

# load pytorch
import torch
import torchvision.transforms.functional as tvf

from writer import Writer
from models.neural_renderer import NeuralRenderer
from models.unet import UNet128
#from models.perceptual_loss import PerceptualLoss

from dataloader.ffhq_pbr import get_dataloader

from torchmetrics.functional.image import peak_signal_noise_ratio

class Trainer(object):
    def __init__(self, exp_name, data_folder, output_folder, configs, pretrained_weights):
        
        # Set the training configs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_folder = data_folder
        #self.batch_size = configs['train']['batch_size']
        self.crop_size = configs['train']['crop_size']
        self.total_epochs = configs['train']['total_epochs']
        self.cross_rate = configs['train']['cross_rate']
        self.num_pixel_samples = configs['train']['num_pixel_samples']
        self.num_light_samples = configs['train']['num_light_samples']

        # Logger setting
        output_path = output_folder + f'/{exp_name}'
        self.weights_dir = output_path + "/weights"
        self.images_dir = output_path + "/images"
        Writer.set_writer(output_path)

        # Data loader setting
        self.train_loader, self.eval_loader = get_dataloader(data_folder=data_folder, 
                                                             eval_rate=0.1, 
                                                             random_seed=0, 
                                                             batch_size=1, 
                                                             shuffle=True, 
                                                             num_workers=4)
        
        # Load model
        neural_render = NeuralRenderer()
        self.neural_render = neural_render.to(self.device)
        
        shadow_estimator = UNet128(in_chns=6, out_chns=1)
        self.shadow_estimator = shadow_estimator.to(self.device)
        
        if pretrained_weights is not None:
            self.neural_render.load_state_dict(pretrained_weights)
        
        # Optimizer setting
        self.adam_optimizer = torch.optim.Adam(params=[{'params': self.neural_render.parameters()},
                                                       {'params': self.shadow_estimator.parameters()}], 
                                               lr=configs['train']['initial_lr'])
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.adam_optimizer, T_max=self.total_epochs)
        
        # Loss function
        self.rec_loss_fn = torch.nn.L1Loss()
        self.shd_loss_fn = torch.nn.MSELoss()
    
    def train(self):
        
        # Initialize training loop
        self.num_epoch = 0
        
        # Main loop
        while True:
            print(f'Epoch: {self.num_epoch}')
            
            # Dynanmically change the learning rate
            # if self.num_epoch >= 50:
            #     self.adjust_learning_rate(self.adam_optimizer, 2e-5)
            # elif self.num_epoch >= 100:
            #     self.adjust_learning_rate(self.adam_optimizer, 5e-6)
            # elif self.num_epoch >= 200:
            #     self.adjust_learning_rate(self.adam_optimizer, 2e-6)
            # elif self.num_epoch >= 300:
            #     self.adjust_learning_rate(self.adam_optimizer, 1e-6)
            
            # Train epoch
            # if self.num_epoch % self.cross_rate == 0:
            #     self.train_epoch(cross=True)
            # else:
            #     self.train_epoch(cross=False)
            
            self.train_epoch(cross=True)
            
            # Evaluation epoch
            with torch.no_grad():
                self.eval_epoch()
            
            if self.num_epoch == self.total_epochs - 1:
                break
            
            self.num_epoch += 1

    def train_epoch(self, cross=False):

        train_iter = iter(self.train_loader)
        
        self.neural_render.train()
        pbar = tqdm(range(len(train_iter)), ncols=80)
        for step in pbar:
            
            # Load training data
            train_data_buffer = next(train_iter)
            
            mask_gt = train_data_buffer['mask_gt'].to(self.device)
            rgb_gt = train_data_buffer['rgb_gt'].to(self.device)
            albedo_gt = train_data_buffer['albedo_gt'].to(self.device)
            roughness_gt = train_data_buffer['roughness_gt'].to(self.device)
            specular_gt = train_data_buffer['specular_gt'].to(self.device)
            normal_gt = train_data_buffer['normal_gt'].to(self.device)
            pos_in_cam_gt = train_data_buffer['pos_in_cam_gt'].to(self.device)
            hdri_gt = train_data_buffer['hdri_gt'].to(self.device)
            
            # Random sampling training pixels
            num_all_training_pixels = rgb_gt[mask_gt].shape[0]
            rand_indices = torch.randperm(num_all_training_pixels, device=self.device)[:self.num_pixel_samples]
            
            # Loss calculation
            # if step % 10 == 0:
            #     render_buffer = {
            #         'rgb_gt': rgb_gt[mask_gt],
            #         'normal_gt': normal_gt[mask_gt],
            #         'albedo_gt': albedo_gt[mask_gt],
            #         'roughness_gt': roughness_gt[mask_gt],
            #         'specular_gt': specular_gt[mask_gt],
            #         'view_pos_gt': view_pos_gt[mask_gt],
            #         'hdri_gt': hdri_gt
            #     }
            # else:
            #     render_buffer = {
            #         'rgb_gt': rgb_gt[mask_gt][rand_indices],
            #         'normal_gt': normal_gt[mask_gt][rand_indices],
            #         'albedo_gt': albedo_gt[mask_gt][rand_indices],
            #         'roughness_gt': roughness_gt[mask_gt][rand_indices],
            #         'specular_gt': specular_gt[mask_gt][rand_indices],
            #         'view_pos_gt': view_pos_gt[mask_gt][rand_indices],
            #         'hdri_gt': hdri_gt
            #     }
            
            if not cross:
                
                render_buffer = {
                    'rgb_gt': rgb_gt[mask_gt][rand_indices],
                    'normal_gt': normal_gt[mask_gt][rand_indices],
                    'albedo_gt': albedo_gt[mask_gt][rand_indices],
                    'roughness_gt': roughness_gt[mask_gt][rand_indices],
                    'specular_gt': specular_gt[mask_gt][rand_indices],
                    'pos_in_cam_gt': pos_in_cam_gt[mask_gt][rand_indices],
                    'hdri_gt': hdri_gt
                }
                
                rgb_rec = self.neural_render(render_buffer=render_buffer, num_light_samples=self.num_light_samples)
                
                # Reconstruction loss
                rec_loss = self.rec_loss_fn(rgb_rec, render_buffer['rgb_gt'])
                
            else:
                with torch.no_grad():
                    
                    render_buffer = {
                        'rgb_gt': rgb_gt[mask_gt],
                        'normal_gt': normal_gt[mask_gt],
                        'albedo_gt': albedo_gt[mask_gt],
                        'roughness_gt': roughness_gt[mask_gt],
                        'specular_gt': specular_gt[mask_gt],
                        'pos_in_cam_gt': pos_in_cam_gt[mask_gt],
                        'hdri_gt': hdri_gt
                    }
                    
                    rgb_rec = self.neural_render(render_buffer=render_buffer, num_light_samples=self.num_light_samples)
            
                B, H, W, C =  rgb_gt.shape
                rec_image = torch.zeros(B, H, W, C, device=self.device)
                rec_image[mask_gt] = rgb_rec

                shadow_map = self.shadow_estimator(torch.cat([rec_image, normal_gt], dim=-1).permute(0,3,1,2)).permute(0,2,3,1)
            
                # Reconstruction loss
                rec_loss = self.rec_loss_fn(rec_image*shadow_map, rgb_gt)

            if rec_loss.item() is not None:
                self.adam_optimizer.zero_grad()
                rec_loss.backward()
                self.adam_optimizer.step()

            # Log loss
            Writer.add_scalar("train/total_loss", rec_loss.item(), step=(step+self.num_epoch*len(self.train_loader)))
            
            # Visualize train result
            # if step == 0:
                
            #     B, H, W, C =  rgb_gt.shape
                
            #     rec_image = torch.zeros(B, H, W, C, device=self.device)
            #     rec_image[mask_gt] = rgb_rec
                
            #     vis_image = torch.cat([albedo_gt[0], 
            #                            roughness_gt[0][...,None].repeat(1,1,3), 
            #                            specular_gt[0][...,None].repeat(1,1,3), 
            #                            (normal_gt[0]+1)/2, 
            #                            rec_image[0], 
            #                            rgb_gt[0]], dim=1)
                
                
            #     tvf.to_pil_image(vis_image.permute(2,0,1)).save(self.images_dir + f"/train_e{self.num_epoch}_s{step}.png")
        
        # Adjust learning rate
        self.lr_scheduler.step()
    
    def eval_epoch(self):
        
        eval_loader = iter(self.eval_loader)
        
        vis_outputs = []
        mse_metric_list = []
        psnr_metric_list = []
        self.neural_render.eval()
        pbar = tqdm(range(100), ncols=80)
        for _ in pbar:
            
            eval_data_buffer = next(eval_loader)
            
            # Input data process
            mask_gt = eval_data_buffer['mask_gt'].to(self.device)
            rgb_gt = eval_data_buffer['rgb_gt'].to(self.device)
            albedo_gt = eval_data_buffer['albedo_gt'].to(self.device)
            roughness_gt = eval_data_buffer['roughness_gt'].to(self.device)
            specular_gt = eval_data_buffer['specular_gt'].to(self.device)
            normal_gt = eval_data_buffer['normal_gt'].to(self.device)
            pos_in_cam_gt = eval_data_buffer['pos_in_cam_gt'].to(self.device)
            hdri_gt = eval_data_buffer['hdri_gt'].to(self.device)
            
            render_buffer = {
                'rgb_gt': rgb_gt[mask_gt],
                'normal_gt': normal_gt[mask_gt],
                'albedo_gt': albedo_gt[mask_gt],
                'roughness_gt': roughness_gt[mask_gt],
                'specular_gt': specular_gt[mask_gt],
                'pos_in_cam_gt': pos_in_cam_gt[mask_gt],
                'hdri_gt': hdri_gt
            }
            
            rgb_rec = self.neural_render(render_buffer=render_buffer, num_light_samples=self.num_light_samples)
            
            # Restore the image resolution
            rec_image = torch.zeros(1,128,128,3).to(self.device)
            rec_image[mask_gt] = rgb_rec
            
            # Shadow map generation
            shadow_map = self.shadow_estimator(torch.cat([rec_image, normal_gt], dim=-1).permute(0,3,1,2)).permute(0,2,3,1)

            # Metrics calculation
            vis_normal = (normal_gt[0] + 1) / 2
            
            vis_image = torch.cat([rgb_gt[0], 
                                   (rec_image*shadow_map)[0], 
                                   shadow_map[0].repeat(1,1,3),
                                   albedo_gt[0], 
                                   roughness_gt[0][...,None].repeat(1,1,3), 
                                   specular_gt[0][...,None].repeat(1,1,3), 
                                   vis_normal], dim=0)
            
            vis_outputs.append(vis_image)
            mse_metric = torch.nn.functional.mse_loss(rgb_rec*shadow_map[mask_gt], rgb_gt[mask_gt])
            psnr_metric = peak_signal_noise_ratio(rgb_rec*shadow_map[mask_gt], rgb_gt[mask_gt])
            
            mse_metric_list.append(mse_metric)
            psnr_metric_list.append(psnr_metric)
        
        # Calculate performance materics
        mean_mse = torch.stack(mse_metric_list).mean()
        mean_psnr = torch.stack(psnr_metric_list, dim=0).mean()
        
        image_outputs = torch.cat(vis_outputs[:5], dim=1)

        Writer.add_scalar("test/MSE", mean_mse, self.num_epoch)
        Writer.add_scalar("test/PSNR", mean_psnr, self.num_epoch)
        
        tvf.to_pil_image(image_outputs.permute(2,0,1)).save(self.images_dir + f"/test_e{self.num_epoch}.png")
        self.neural_render.save_model(weights_dir=self.weights_dir)
        torch.save(self.shadow_estimator.state_dict(), self.weights_dir + '/ShadowEstimator.pth')

    # Helper function
    def adjust_learning_rate(self, optimizer, lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        #lr = base_lr * (0.1 ** (epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
