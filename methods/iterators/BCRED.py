from ..optim_method import Method
from ..block_optim_method import BlockMethod
import torch
from deepinv.optim.prior import PnP
from deepinv.models import DnCNN

class BCRED(BlockMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.prior = PnP(denoiser=DnCNN(pretrained="download", device=self.device))
        # self.prior = PnP(denoiser=DnCNN(pretrained="download_lipschitz", device=self.device))

        self.lamb = 1.
        if self.fidelity=='cauchy':
            self.lamb = 1e3
            self.tau = [1.5/(self.data_fidelity.gamma+2*self.lamb)]*self.nblocks
        else:
            self.tau = [2/(1+2*self.lamb)]*self.nblocks

        self.early_stopping = False

    def restricted_denoiser(self, x, sigma):
        if self.Ui is not None:
            x = self.Ui.A_extended(x)

        Dx = self.prior.prox(x, sigma)

        if self.Ui is not None:
            Dx = self.Ui.trim(Dx)
            Dx = self.Ui(Dx)

        return Dx

                    
    def iteration(self, x, y, physics, x_true, filename):

        x_old = x

        
        z = x_old

        for itr in range(1,self.maxitr+1):

            

            Dg = self.lamb*(self.Uitilde(x)-self.restricted_denoiser(x, sigma=self.sigma_denoiser)) + self.Uitilde(self.data_fidelity.grad(x,y,physics))
            x_old = x_old - self.Uitilde(x_old - x)

            x = x - self.tau[self.block_counter]*Dg

            self.update_block()

            if self.writer is not None:
                self.save_metrics(x, z, x_old, y, x_true, physics, filename, itr)
            if self.early_stopping:
                if self.check_convergence(x, x_old, y, physics):
                    print(f'Convergence reached at iteration {itr} for {filename}')
                    break
                
        z = self.last_step(x)
        return x, z
        
    