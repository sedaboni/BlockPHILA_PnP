from .optim_method import Method
import torch
from utils.masker import Uitilde, Ui
import cv2

class BlockMethod(Method):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tau = [self.tau]*self.nblocks
        self.Uitilde = Uitilde(img_size=self.img_size)
        self.Ui = Ui(img_size=self.img_size)

        self.init_div()
        self.init_path_size()

        self.offset = 16 #8
        self._bb_warmup = self.nblocks

    def update_block(self, reset=False):
        self.update_block_counter(reset)
        self.set_block()
        self.set_offsets()
        self.set_mask()
    
    def update_block_counter(self, reset=False):
        if reset:
            self.block_counter = 0
        else:
            self.block_counter = (self.block_counter + 1)%self.nblocks
    
    def set_mask(self):
        self.Uitilde.set_mask(self.h_start, self.w_start, self.h_end, self.w_end)
        self.Ui.set_mask(self.h_start, self.w_start, self.h_end, self.w_end)
        self.Ui.set_offsets(self.up, self.down, self.left, self.right)

    def init_div(self):
        if self.nblocks==1:
            self.div_h = 1
        if self.nblocks==2 or self.nblocks==4:
            self.div_h = 2
        if self.nblocks==8 or self.nblocks==16:
            self.div_h = 4

        self.div_w = self.nblocks//self.div_h

    def init_path_size(self):
        self.patch_size_h = self.img_size[-2]//self.div_h
        self.patch_size_w = self.img_size[-1]//self.div_w
    
    def set_block(self):
        q = self.block_counter//self.div_h
        r = self.block_counter%self.div_h
        if self.nblocks==4:
            q, r = r, q
            
        self.h_start = self.patch_size_h*r
        self.h_end = self.patch_size_h*(r+1)

        self.w_start = self.patch_size_w*q
        self.w_end = self.patch_size_w*(q+1)

    
    def set_offsets(self):
        self.up = (self.h_start>0)*self.offset
        self.down = (self.h_end<self.img_size[-2])*self.offset

        self.left = (self.w_start>0)*self.offset
        self.right = (self.w_end<self.img_size[-1])*self.offset
    
    def set_state(self):
        super().set_state()
        self.update_block(reset=True)

    def last_step(self, x):
        z = x
        for _ in range(self.nblocks):
            z = z - self.Uitilde(self.denoiser.potential_grad(x, self.sigma_denoiser, Ui=self.Ui))
            self.update_block()
        return z

    def _current_tau(self):
        return self.tau[self.block_counter]
    
    def _bb_project(self, v):
        return self.Uitilde(v)

    def _set_tau(self, value):
        self.tau[self.block_counter] = value

    def _reset_tau(self):
        for i in range(self.nblocks):
            self.tau[i] = self.tau_init

    def get_i(self, itr):
        return (itr-1)//self.nblocks