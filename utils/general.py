import torch

def split_model_inputs(input, total_pixels, split_size):
    '''
    Split the input to fit Cuda memory for large resolution.
    Can decrease the value of split_num in case of cuda out of memory error.
    '''
    split_size = split_size                                                                                                            # [S]
    split_input = []
    split_indexes = torch.split(torch.arange(total_pixels).cuda(), split_size, dim=0)
    for indexes in split_indexes:
        data = {}
        data['positions'] = torch.index_select(input['positions'], 1, indexes)
        data['normals'] = torch.index_select(input['normals'], 1, indexes)
        data['view_dirs'] = torch.index_select(input['view_dirs'], 1, indexes)
        data['textures'] = torch.index_select(input['textures'], 1, indexes)
        data['masks'] = torch.index_select(input['masks'], 1, indexes)
        split_input.append(data)
        
    return split_input

def hdr2ldr(img, scale=0.666667):
    #img = img * scale
    # img = 1 - np.exp(-3.0543 * img)  # Filmic
    img = (img * (2.51 * img + 0.03)) / (img * (2.43 * img + 0.59) + 0.14)  # ACES
    return img