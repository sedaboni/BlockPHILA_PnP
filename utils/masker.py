import torch
import deepinv as dinv

class Uitilde(dinv.physics.LinearPhysics):

    def __init__(self, img_size, **kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
    
    def set_mask(self, h_start=None, w_start=None, h_end=None, w_end=None):
        self.h_start, self.w_start = h_start, w_start
        self.h_end, self.w_end = h_end, w_end
    
    def A(self, y):
        x = torch.zeros_like(y)
        x[:,:,self.h_start:self.h_end,self.w_start:self.w_end]  = y[:,:,self.h_start:self.h_end,self.w_start:self.w_end]
        return x

    def A_adjoint(self, y):
        x = torch.zeros_like(y)
        x[:,:,self.h_start:self.h_end,self.w_start:self.w_end]  = y[:,:,self.h_start:self.h_end,self.w_start:self.w_end]
        return x
    
class Ui(dinv.physics.LinearPhysics):

    def __init__(self, img_size, **kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
    
    def set_mask(self, h_start=None, w_start=None, h_end=None, w_end=None):
        self.h_start, self.w_start = h_start, w_start
        self.h_end, self.w_end = h_end, w_end

    def set_offsets(self, up=0, down=0, left=0, right=0):
        self.up, self.down = up, down
        self.left, self.right = left, right
    
    def A(self, y):
        x = torch.zeros(y.size(0), self.img_size[0], self.img_size[1], self.img_size[2]).type_as(y)
        x[:,:,self.h_start:self.h_end,self.w_start:self.w_end]  = y
        return x

    def A_adjoint(self, y):
        return y[:,:,self.h_start:self.h_end,self.w_start:self.w_end]
    
    def A_extended(self, y):
        return y[:,:,self.h_start-self.up:self.h_end+self.down,(self.w_start-self.left):(self.w_end+self.right)]
    
    def trim(self,y):
        return y[:,:,self.up:(self.h_end-self.h_start)+self.up,self.left:(self.w_end-self.w_start)+self.left]