import os
import warnings
warnings.filterwarnings('ignore')

import argparse
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import json

from trainer import Trainer
from writer import Writer

def main(exp_name, data_folder, output_folder, config_path, weight_path):
    
    # Prase config file
    with open(config_path, 'r') as f:
        configs = json.load(f)
    
    if weight_path is not None:
        print(f"Load pretrained weights from: {weight_path}")
        pretrained_weights = torch.load(os.path.join(weight_path, 'NeuralRenderer.pth'))
        pretrained_weights = {k: v for k, v in pretrained_weights.items() if "unet" not in k}
    else:
        pretrained_weights = None
    
    trainer = Trainer(exp_name=exp_name,
                      data_folder=data_folder,
                      output_folder=output_folder,
                      configs=configs,
                      pretrained_weights=pretrained_weights)

    trainer.train()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    # train options
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--data_folder', default='./dataset/ffhq256_pbr', type=str)
    parser.add_argument('--output_folder', default='./output', type=str)
    parser.add_argument('--config_path', default='./configs/configs.json', type=str)
    parser.add_argument('--weight_path', default=None, type=str)
    
    args = parser.parse_args()
    
    main(exp_name=args.exp_name, 
         data_folder=args.data_folder, 
         output_folder=args.output_folder,
         config_path=args.config_path,
         weight_path=args.weight_path)
