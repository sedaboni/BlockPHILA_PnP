import os
import torch
import cv2 as cv
import numpy as np
import deepinv as dinv
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, path, physics=None, grayscale=False, device="cpu"):
        super().__init__()

        self.device = device
        self.folder_im = os.path.join('data',path)
        self.file_names_im = sorted(os.listdir(self.folder_im))
        self.file_path_list = [os.path.join(self.folder_im, i) for i in self.file_names_im]

        self.physics = physics

        if grayscale:
            self.n_channels = 1
        else:
            self.n_channels = 3
        

    def __len__(self):
        return len(self.file_names_im)
    
    def __getitem__(self, idx):
        filename = os.path.splitext(self.file_names_im[idx])[0]
        im_true = self.imread_uint(self.file_path_list[idx])
        
        if self.physics is not None:
            degraded_im = self.physics(im_true.unsqueeze(0)).squeeze(0) 
            return im_true, degraded_im, filename
        else:
            return im_true, filename
    
    def set_physics(self, physics):
        self.physics = physics
    
    def imread_uint(self, path):
        #  input: path
        # output: HxWx3(RGB or GGG), or HxWx1 (G)
        if self.n_channels == 1:
            img = cv.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
            img = np.expand_dims(img, axis=2)  # HxWx1
        elif self.n_channels == 3:
            img = cv.imread(path, cv.IMREAD_UNCHANGED)  # BGR or G
            if img.ndim == 2:
                img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)  # GGG
            else:
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # RGB

        img = np.float32(img / 255.)
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img.to(self.device)
    
    def get_img_size(self):
        if self.physics is None:
            im_true, _ = self.__getitem__(0)
        else:
            im_true, _, _ = self.__getitem__(0)
        
        return im_true.size()[-3:]