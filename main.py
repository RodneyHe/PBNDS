import os
import warnings
warnings.filterwarnings('ignore')

import argparse
import torch
from trainer import Trainer
from writer import Writer

def main(exp_name, data_folder, output_folder, weight_path):
    
    # Set the training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if weight_path is not None:
        pretrained_weights = torch.load(os.path.join(weight_path, 'NeuralRenderer.pth'))
    else:
        pretrained_weights = None
    
    trainer = Trainer(exp_name=exp_name,
                      data_folder=data_folder,
                      output_folder=output_folder,
                      device=device,
                      pretrained_weights=pretrained_weights)

    trainer.train()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    # train options
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--weight_path', default=None, type=str)
    
    args = parser.parse_args()
    
    main(exp_name=args.exp_name, 
         data_folder=args.data_folder, 
         output_folder=args.output_folder,
         weight_path=args.weight_path)