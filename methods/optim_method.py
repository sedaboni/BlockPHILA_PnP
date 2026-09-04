import deepinv as dinv
import torch
from utils.GSPnP import GSDRUNet
from utils.fidelity import L2
from utils.cauchy_fidelity import CauchyFidelity
from utils.utils import shift_pixel
import cv2
import time

class Method(torch.nn.Module):
    def __init__(self, lamb, tau, sigma_denoiser, img_size, nblocks=1, prox_mode='CG', maxitr=200, early_stopping=True, crit_conv='residual', fidelity='L2', thres_conv=1e-5, gamma_cauchy=1e-2, writer=None, grayscale=False, device="cpu", den_type='drunet', BB=None, name=None):
        super(Method, self).__init__()
        self.name = name or type(self).__name__
        self.device = device

        if grayscale:
            self.n_channels = 1
        else:
            self.n_channels = 3

        self.denoiser = GSDRUNet(
            in_channels=self.n_channels,
            out_channels=self.n_channels,
            pretrained='download',  # automatically downloads the pretrained weights, set to a path to use custom weights.
            device=device,
            den_type=den_type
        )

        self.fidelity = fidelity

        if fidelity=='L2':
            self.data_fidelity = L2(mode=prox_mode)
        elif fidelity=='cauchy':
            self.data_fidelity = CauchyFidelity(gamma=gamma_cauchy)
        

        self.lamb = lamb
        self.tau = tau
        self.tau_init = tau
        self.sigma_denoiser = sigma_denoiser
        self.nblocks = nblocks
        self.maxitr = maxitr
        self.early_stopping = early_stopping
        self.crit_conv = crit_conv
        self.thres_conv = thres_conv
        self.img_size = img_size
        
        self.writer = writer

        self.block_counter = 0
        self._bb_warmup = 1
        self.BB = BB
        self.metric = False
        self.bb_min, self.bb_max = 1e-2, 1e3

        self.PSNR = dinv.loss.metric.PSNR()
        self.SSIM = dinv.loss.metric.SSIM()
    
    def forward(self, y, physics, x_true, filename):

        self.set_state()

        with torch.no_grad():
            
            x0 = self.prepare_init(y, physics)
            self.norm_x0 = torch.sum(x0 ** 2)

            self.save_metrics(x0, x0, x0, y, x_true, physics, filename,0)

            x, z = self.iteration(x0, y, physics, x_true, filename)
        
        self.save_last(z, x_true, filename)
        
        return x, z
    
    def iteration(self, x, y, physics, x_true, filename):
        pass
        
    def set_state(self):
        self.f = None
        self.g = None
        self.F = None
        self.f_old = None
        self.g_old = None
        self.F_old = None

        self.start_time = time.time()
    
    def save_last(self, z, x_true, filename):
        PSNR_x = self.PSNR(z, x_true)
        self.writer.add_scalar(f'PSNR last/{filename}', PSNR_x.item(), 0)


        SSIM_x = self.SSIM(z, x_true)
        self.writer.add_scalar(f'SSIM last/{filename}', SSIM_x.item(), 0)
    
    def save_metrics(self, x, z, x_old, y, x_true, physics, filename, itr):

        with torch.no_grad():
            
            if (self.f is not None) and (self.g is not None) and (self.F is not None):
                self.f_old = self.f
                self.g_old = self.g
                self.F_old = self.F
                
            self.f = self.data_fidelity(x, y, physics)
            self.g = self.lamb*self.denoiser.potential(x, sigma=self.sigma_denoiser)
            self.F = self.f + self.g
            
            self.writer.add_scalar(f'f/{filename}', self.f.item(), itr)
            self.writer.add_scalar(f'g/{filename}', self.g.item(), itr)
            self.writer.add_scalar(f'F/{filename}', self.F.item(), itr)


            self.writer.add_scalar(f'max x/{filename}', x.max().item(), itr)
            self.writer.add_scalar(f'min x/{filename}', x.min().item(), itr)

            PSNR_x = self.PSNR(x, x_true)
            self.writer.add_scalar(f'PSNR x/{filename}', PSNR_x.item(), itr)


            SSIM_x = self.SSIM(x, x_true)
            self.writer.add_scalar(f'SSIM x/{filename}', SSIM_x.item(), itr)

            if z is not None:
                PSNR_z = self.PSNR(z, x_true)
                self.writer.add_scalar(f'PSNR z/{filename}', PSNR_z.item(), itr)
                SSIM_z = self.SSIM(z, x_true)
                self.writer.add_scalar(f'SSIM z/{filename}', SSIM_z.item(), itr)

            conv = torch.sum((x - x_old) ** 2) / self.norm_x0
            self.writer.add_scalar(f'conv/{filename}', conv.item(), itr)
            self.writer.add_scalar(f'tau/{filename}', self._current_tau(), itr)

            if hasattr(self, 'counter'):
                self.writer.add_scalar(f'ls_itr/{filename}', self.counter, itr)
            
            if hasattr(self, 'Ds'):
                self.writer.add_scalar(f'max D/{filename}', self.Ds.max().item(), itr)
                self.writer.add_scalar(f'min D/{filename}', self.Ds.min().item(), itr)



    def check_convergence(self, x, x_old, y, physics):
        if self.crit_conv == 'residual':
            conv = torch.norm(x - x_old) / torch.norm(x_old)
            condition = conv < self.thres_conv
        elif self.crit_conv == 'cost':
            if (self.f_old is None) and (self.g_old is None) and (self.F_old is None):
                self.f_old = self.data_fidelity(x_old, y, physics)
                self.g_old = self.denoiser.potential(x_old, sigma=self.sigma_denoiser)
                self.F_old = self.f_old + self.lamb * self.g_old
            conv = torch.abs(self.F - self.F_old) / torch.abs(self.F_old)
            condition = conv < self.thres_conv
        elif self.crit_conv == 'time':
            condition = (time.time() - self.start_time) > 2.
        else:
            raise ValueError(f"crit_conv {self.crit_conv!r} not implemented")
        
        return condition

    
    def prepare_up(self, y, physics):
        temp = cv2.resize(y.squeeze(0).permute(1, 2, 0).cpu().numpy(), (physics.factor*y.size(-2), physics.factor*y.size(-1)),interpolation=cv2.INTER_CUBIC)
        if physics.factor>1:
            temp = shift_pixel(temp, physics.factor)
            x = torch.tensor(temp).permute(2, 1, 0).unsqueeze(0).to(physics.device).contiguous()
        else:
            x = torch.tensor(temp).permute(2, 0, 1).unsqueeze(0).to(physics.device).contiguous()
        return x
    
    def prepare_init(self, y, physics):

        x = self.prepare_up(y,physics)

        if self.fidelity=='L2':
            if isinstance(self.tau, (list, tuple)):
                x = self.data_fidelity.prox(x,y,physics,gamma=self.tau[0])
            else:
                x = self.data_fidelity.prox(x,y,physics,gamma=self.tau)
        
        if self.fidelity=='cauchy':
            x = physics.A_adjoint(y)
        return x
    
    def last_step(self, x):
        z = x - self.denoiser.potential_grad(x, self.sigma_denoiser)
        return z
    
    def _current_tau(self):
        return self.tau
    
    def _bb_project(self, v): # identity for the full-image case
        return v

    def _set_tau(self, value):
        self.tau = value
    
    def _reset_tau(self):
        self.tau = self.tau_init
    
    def init_bb(self, x):
        """Call once at the top of iteration()."""
        if self.BB is not None:
            self._reset_tau()
            self._Dg_old = torch.zeros_like(x)
    
    def _bb_stepsize(self, sk, gk):
        if self.BB == 'BB1':
            tau = sk.norm()**2 / (sk * gk).sum()
        elif self.BB == 'BB2':
            tau = (sk * gk).sum() / gk.norm()**2
        elif self.BB == 'geo':
            tau = sk.norm() / gk.norm()
        elif self.BB == 'geo_vm':
            tau = torch.norm(sk/torch.sqrt(self.Ds)) / torch.norm(gk*torch.sqrt(self.Ds))
        else:
            raise ValueError(f"unknown BB rule {self.BB!r}")
        if tau.item()<=0 or torch.isnan(tau):
            tau = self.tau_init
        else:
            tau = torch.clamp(tau, min=self.bb_min, max=self.bb_max).item()
        return tau
    
    def update_bb_step(self, x, x_old, Dg, itr):
        """Call once per iteration where the BB block used to be. No-op if BB disabled."""
        if self.BB is not None:
            if itr > self._bb_warmup:
                sk = self._bb_project(x - x_old)
                gk = self.lamb * self._bb_project(Dg - self._Dg_old) 
                self._set_tau(self._bb_stepsize(sk, gk))
            self._Dg_old = self._Dg_old - self._bb_project(self._Dg_old - Dg)

    def init_Ds(self, x):
        self.t1 = 1e1
        self.t2 = 1.1
        self.Sk = torch.zeros_like(x)
        self.Svark = torch.zeros_like(x)

        self.beta1 = 0.9
        self.beta2 = 0.999

    def get_i(self, itr):
        return itr-1

    def update_Ds(self, Dg, itr):
        i = self.get_i(itr)
   
        self.Sk = self.Sk + torch.pow(Dg, 2)
        self.Sk_hat = self.lamb*self.Sk**(1/2)
        if itr > self._bb_warmup:
            self.D = 1/(self.Sk_hat+1e-8)
            self.D_upp = (1+self.t1/(i+1)**(1+self.t2))
            self.D_low = 1/(self.D_upp)
            self.Ds = self.D_upp*self.D/self.D.max()
            self.Ds = torch.clamp(self.Ds, min=self.D_low, max=self.D_upp)
        else:
            self.Ds = torch.ones_like(Dg)