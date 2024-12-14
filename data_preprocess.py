import os, json, argparse
from tqdm import tqdm
import torch
import numpy as np
from skimage.transform import estimate_transform, warp
import cv2 as cv

import warnings
warnings.filterwarnings('ignore')

from models.decalib.deca import DECA
from models.decalib.utils.config import cfg as deca_cfg
from models.decalib.datasets.detectors import FAN

def preprocess(data_folder):
    
    # Load FLAME model and DECA model
    deca_cfg['model']['flame_model_path'] = './pretrained/generic_model.pkl'
    deca_cfg['pretrained_modelpath'] = './pretrained/deca_model.tar'
    deca_cfg['model']['flame_lmk_embedding_path'] = './pretrained/landmark_embedding.npy'
    deca_cfg['model']['use_tex'] = False
    
    deca = DECA(config=deca_cfg)
    face_detector = FAN()
    
    rgb_gt_path = data_folder + f'/bgremoval'
    rgb_subfolder_list = sorted(os.listdir(rgb_gt_path))
        
    gt_indices_list = []
    for sub_folder in rgb_subfolder_list:
        for name in sorted(os.listdir(os.path.join(rgb_gt_path, sub_folder))):
            gt_indices_list.append(os.path.join(name[:-4]))
    
    pred_fov_dict = {}
    pbar = tqdm(range(len(gt_indices_list)), ncols=80)
    for i in pbar:
        subfolder = int(gt_indices_list[i]) // 1000 * 1000
        image = cv.imread(os.path.join(rgb_gt_path, f'{subfolder:05}/{gt_indices_list[i]}.png'), \
                          cv.IMREAD_UNCHANGED)
        
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                alpha_channel = image[...,3]
                bgr_channels = image[...,:3]
                rgb_channels = cv.cvtColor(bgr_channels, cv.COLOR_BGR2RGB)
                
                # White Background Image
                background_image = np.zeros_like(rgb_channels, dtype=np.uint8)
                
                # Alpha factor
                alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) / 255.
                alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)

                # Transparent Image Rendered on White Background
                base = rgb_channels * alpha_factor
                background = background_image * (1 - alpha_factor)
                image = base + background
            else:
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
        bbox, bbox_type = face_detector.run(image)

        if len(bbox) < 4:
            pred_fov_dict[str(gt_indices_list[i])] = 85
        else:
            left = bbox[0]; right=bbox[2]
            top = bbox[1]; bottom=bbox[3]
        
            old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
            size = int(old_size*1.25)
            
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
            
            DST_PTS = np.array([[0, 0], [0, 223], [223, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)
            
            image = image / 255.

            dst_image = warp(image, tform.inverse, output_shape=(224, 224))
            dst_image = dst_image.transpose(2,0,1)
            
            codedict = deca.encode(torch.tensor(dst_image).float().cuda()[None])
            
            pred_fov = torch.arctan(image.shape[1]/(2*codedict['cam'][0, 0])) / torch.pi * 180
            
            pred_fov_dict[str(gt_indices_list[i])] = pred_fov.item()
        
    json_obj = json.dumps(pred_fov_dict, indent=4)
    
    with open(os.path.join(data_folder, 'pred_fov.json'), 'w') as outfile:
        outfile.write(json_obj)
    
def bbox2point(left, right, top, bottom, type='bbox'):
    ''' bbox from detector and landmarks are different
    '''
    if type=='kpt68':
        old_size = (right - left + bottom - top)/2*1.1
        center = torch.tensor([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
    elif type=='bbox':
        old_size = (right - left + bottom - top)/2
        center = torch.tensor([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
    else:
        raise NotImplementedError
    return old_size, center

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str)
    args = parser.parse_args()
    
    preprocess(data_folder=args.data_folder)
