import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import json
import argparse
from tqdm import tqdm

# load pytorch
import torch
import torchvision.transforms.functional as tvf
from torch.utils.data.dataloader import DataLoader

from writer import Writer
from models.neural_renderer import NeuralRenderer
from models.perceptual_loss import PerceptualLoss

from dataloader.ffhq_pbr import FFHQPBR
from utils import general

from torchmetrics.functional.image import peak_signal_noise_ratio

class Trainer(object):
    def __init__(self, exp_name, data_folder, output_folder, device, pretrained_weights=None):
        self.device = device
        self.data_folder = data_folder
        self.batch_size = 1
        self.sample_num = 4096

        # Logger setting
        output_path = output_folder + f'/{exp_name}'
        self.weights_dir = output_path + "/weights"
        self.images_dir = output_path + "/images"
        Writer.set_writer(output_path)

        # Data loader setting
        train_dataset = FFHQPBR(data_path=data_folder, mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                       shuffle=True, num_workers=4, pin_memory=True)
        
        eval_dataset = FFHQPBR(data_path=data_folder, mode='eval')
        self.eval_loader = DataLoader(eval_dataset, batch_size=1, 
                                      shuffle=True, num_workers=4, pin_memory=True)
        
        # Load model
        neural_render = NeuralRenderer()
        self.neural_render = neural_render.to(device)
        
        if pretrained_weights is not None:
            self.neural_render.load_state_dict(pretrained_weights)
        
        # Optimizer setting
        self.adam_optimizer = torch.optim.Adam(params=self.neural_render.parameters(), lr=5e-5)
        
        # Loss function
        self.rec_loss_fn = torch.nn.L1Loss()
        self.perc_loss_fn = PerceptualLoss()

        # Training setting
        self.num_epoch = 0
        self.total_epochs = 3000
        self.num_samples = 128
    
    def train(self):
        
        # Main loop
        while True:
            print(f'Epoch: {self.num_epoch}')
            
            # Dynanmically change the learning rate
            if self.num_epoch >= 50:
                self.adjust_learning_rate(self.shader_optimizer, 2e-5)
            elif self.num_epoch >= 100:
                self.adjust_learning_rate(self.shader_optimizer, 5e-6)
            elif self.num_epoch >= 200:
                self.adjust_learning_rate(self.shader_optimizer, 2e-6)
            elif self.num_epoch >= 300:
                self.adjust_learning_rate(self.shader_optimizer, 1e-6)
            
            # Train epoch
            self.train_epoch()
            
            # Evaluation epoch
            with torch.no_grad():
                self.eval_epoch()
            
            if self.num_epoch == self.total_epochs - 1:
                break
            
            self.num_epoch += 1

    def train_epoch(self):

        train_iter = iter(self.train_loader)
        
        self.neural_render.train()
        pbar = tqdm(range(len(train_iter)), ncols=80)
        for step in pbar:

            train_data_buffer = next(train_iter)
            
            # Random sampling pixels
            mask_gt = train_data_buffer['mask_gt'].to(self.device)
            rgb_gt = train_data_buffer['rgb_gt'].to(self.device)
            albedo_gt = train_data_buffer['albedo_gt'].to(self.device)
            roughness_gt = train_data_buffer['roughness_gt'].to(self.device)
            specular_gt = train_data_buffer['specular_gt'].to(self.device)
            normal_gt = train_data_buffer['normal_gt'].to(self.device)
            
            num_all_training_pixels = rgb_gt[mask_gt].shape[0]
            rand_indices = torch.randperm(num_all_training_pixels)[:self.sample_num].to(self.device)
            
            # Loss calculation
            if step % 10 == 0:
                render_buffer = {
                    'rgb_gt': rgb_gt[mask_gt],
                    'normal_gt': normal_gt[mask_gt],
                    'albedo_gt': albedo_gt[mask_gt],
                    'roughness_gt': roughness_gt[mask_gt],
                    'specular_gt': specular_gt[mask_gt],
                    'view_pos_gt': train_data_buffer['view_pos_gt'].to(self.device)[mask_gt],
                    'hdri_gt': train_data_buffer['hdri_gt'].to(self.device)
                }
            else:
                render_buffer = {
                    'rgb_gt': rgb_gt[mask_gt][rand_indices],
                    'normal_gt': normal_gt[mask_gt][rand_indices],
                    'albedo_gt': albedo_gt[mask_gt][rand_indices],
                    'roughness_gt': roughness_gt[mask_gt][rand_indices],
                    'specular_gt': specular_gt[mask_gt][rand_indices],
                    'view_pos_gt': train_data_buffer['view_pos_gt'].to(self.device)[mask_gt][rand_indices],
                    'hdri_gt': train_data_buffer['hdri_gt'].to(self.device)
                }
            
            rgb_rec = self.neural_render(render_buffer=render_buffer, num_samples=self.num_samples)
            
            # Reconstruction loss
            rec_loss = self.rec_loss_fn(rgb_rec, render_buffer['rgb_gt'])
            
            # Perceptual loss
            if step % 10 == 0:
                rec_image = torch.zeros(1,128,128,3).to(self.device)
                rec_image[mask_gt] = rgb_rec
                
                perc_loss = self.perc_loss_fn(rec_image.permute(0,3,1,2), 
                                              rgb_gt.permute(0,3,1,2), 
                                              mask_gt[...,None].permute(0,3,1,2), 
                                              layers=[2])
                
            else:
                perc_loss = torch.tensor([0.]).to(self.device)
            
            total_loss = rec_loss + perc_loss

            if rec_loss.item() != 0:
                self.adam_optimizer.zero_grad()
                total_loss.backward()
                self.adam_optimizer.step()

            # Log loss
            Writer.add_scalar("train/total_loss", total_loss.item(), step=(step+self.num_epoch*len(self.train_loader)))
            Writer.add_scalar("train/rec_loss", rec_loss.item(), step=(step+self.num_epoch*len(self.train_loader)))
            Writer.add_scalar("train/perc_loss", perc_loss.item(), step=(step+self.num_epoch*len(self.train_loader)))
            
            # Visualize train result
            if step == 0:
                
                vis_normal = (normal_gt[0] + 1) / 2
                
                vis_image = torch.cat([albedo_gt[0], 
                                       roughness_gt[0][...,None].repeat(1,1,3), 
                                       specular_gt[0][...,None].repeat(1,1,3), 
                                       vis_normal, 
                                       rec_image[0], 
                                       rgb_gt[0]], dim=1)
                
                
                tvf.to_pil_image(vis_image.permute(2,0,1)).save(self.images_dir + f"/train_e{self.num_epoch}_s{step}.png")
    
    def eval_epoch(self):
        
        eval_loader = iter(self.eval_loader)
        
        vis_outputs = []
        mse_metric_list = []
        psnr_metric_list = []
        self.neural_render.eval()
        pbar = tqdm(range(50), ncols=80)
        for _ in pbar:
            
            eval_data_buffer = next(eval_loader)
            
            # Input data process
            mask_gt = eval_data_buffer['mask_gt'].to(self.device)
            rgb_gt = eval_data_buffer['rgb_gt'].to(self.device)
            albedo_gt = eval_data_buffer['albedo_gt'].to(self.device)
            roughness_gt = eval_data_buffer['roughness_gt'].to(self.device)
            specular_gt = eval_data_buffer['specular_gt'].to(self.device)
            normal_gt = eval_data_buffer['normal_gt'].to(self.device)
            
            render_buffer = {
                'rgb_gt': rgb_gt[mask_gt],
                'normal_gt': normal_gt[mask_gt],
                'albedo_gt': albedo_gt[mask_gt],
                'roughness_gt': roughness_gt[mask_gt],
                'specular_gt': specular_gt[mask_gt],
                'view_pos_gt': eval_data_buffer['view_pos_gt'].to(self.device)[mask_gt],
                'hdri_gt': eval_data_buffer['hdri_gt'].to(self.device)
            }
            
            rgb_rec = self.neural_render(render_buffer=render_buffer, num_samples=self.num_samples)
            
            # Restore the image resolution
            rec_image = torch.zeros(1,128,128,3).to(self.device)
            rec_image[mask_gt] = rgb_rec

            # Metrics calculation
            vis_normal = (normal_gt[0] + 1) / 2
            
            vis_image = torch.cat([rgb_gt[0], 
                                   rec_image[0], 
                                   albedo_gt[0], 
                                   roughness_gt[0][...,None].repeat(1,1,3), 
                                   specular_gt[0][...,None].repeat(1,1,3), 
                                   vis_normal], dim=0)
            
            vis_outputs.append(vis_image)
            mse_metric = torch.nn.functional.mse_loss(rgb_rec, rgb_gt[mask_gt])
            psnr_metric = peak_signal_noise_ratio(rgb_rec, rgb_gt[mask_gt])
            
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

    # Helper function
    def adjust_learning_rate(self, optimizer, lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        #lr = base_lr * (0.1 ** (epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
