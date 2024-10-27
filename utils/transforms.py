import numpy as np
import torch
from typing import Tuple

# pytorch3d to blender
def P2B(R, T):
    
    P2B_R1 = torch.tensor([[1, 0,  0],
                           [0, 0, -1],
                           [0, 1,  0]], dtype=torch.float32).to(R.device)
    
    P2B_R2 = torch.tensor([[-1, 0, 0],
                           [ 0, 1, 0],
                           [ 0, 0,-1]], dtype=torch.float32).to(R.device)
    
    P2B_T  = torch.tensor([[-1, 0, 0],
                           [ 0, 0, 1],
                           [ 0,-1, 0]], dtype=torch.float32).to(R.device)
    
    vec4w  = torch.tensor([[0,0,0,1]], dtype=torch.float32).to(R.device)
    
    Bcol3 = P2B_T @ R @ T
    B3x3  = P2B_R1 @ R @ P2B_R2
    B3x4 = torch.cat([B3x3, Bcol3[:, None]], axis=1)
    
    B = torch.cat([B3x4, vec4w], axis=0)
    
    return B

# blender to pytorch3d
def B2P(B):
    '''
    The defualt world coordinate convention for this transformation is Y Up, -Z Forward, X Left.
    This is the defult obj file export setting in blender. So all model coordinates follow this convention
    in the experiment.
    '''
    
    B2P_R1 = torch.tensor([[1,  0, 0],
                           [0,  0, 1],
                           [0, -1, 0]], dtype=torch.float32).to(B.device)
    
    # B2P_R2 is to transfrom 
    B2P_R2 = torch.tensor([[-1, 0, 0],
                           [ 0, 1, 0],
                           [ 0, 0,-1]], dtype=torch.float32).to(B.device)
    
    B2P_T  = torch.tensor([[-1, 0, 0],
                           [ 0, 0,-1],
                           [ 0, 1, 0]], dtype=torch.float32).to(B.device)
    
    vec4w  = torch.tensor([[0,0,0,1]], dtype=torch.float32).to(B.device)
    
    R = B2P_R1 @ B[:3, :3] @ B2P_R2
    T = B2P_T @ B[:3, 3] @ R
    
    RT_3x4 = torch.cat([R, T[:, None]], axis=1)
    RT_4x4 = torch.cat([RT_3x4, vec4w], axis=0)
    
    return R, T, RT_4x4
