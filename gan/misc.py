import  pytorch_fid_wrapper as pfw
import numpy as np
import torch
import torch.nn as nn
import os
import torch.nn as nn
import pickle

def ortho(model, strength=1e-4, blacklist=[]):
    with torch.no_grad():
        for param in model.parameters():
            # Only apply this to parameters with at least 2 axes, and not in the blacklist
            if param.grad is None or len(param.shape) < 2 or any([param is item for item in blacklist]):
                continue
            w = param.view(param.shape[0], -1)
            grad = (2 * torch.mm(torch.mm(w, w.t()) 
                    * (1. - torch.eye(w.shape[0], device=w.device)), w))
            param.grad.data += strength * grad.view(param.shape)

class RandomApplyEach(nn.Module):
    def __init__(self, transforms, p):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, img):
        for t in self.transforms:
            if self.p > torch.rand(1, device='cuda'):
                img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ImageDatasetWrapper():
    def __init__(self, dataset):
        self.dataset=dataset

    def __getitem__(self, key):
        if isinstance(key, slice):
            range(*key.indices(len(self.dataset)))
            return torch.tensor([np.asarray(self.dataset[i][0]) for i in range(*key.indices(len(self.dataset)))])
        elif isinstance(key, int):
            return torch.tensor(self.dataset[key][0])

    def __len__(self):
        return len(self.dataset)

def setup_fid(image_dataset, var_dir, batch_size, device):
    pfw.set_config(batch_size=batch_size, device=device)
    if os.path.isfile(var_dir+'fid_stats.pkl'):
        with open(var_dir+'fid_stats.pkl', 'rb') as f:
            fid_real_m, fid_real_s = pickle.load(f)
    else:
        fid_real_m, fid_real_s = pfw.get_stats(ImageDatasetWrapper(image_dataset))
        with open(var_dir+'fid_stats.pkl', 'wb') as f:
            pickle.dump([fid_real_m, fid_real_s], f)
    return fid_real_m, fid_real_s
