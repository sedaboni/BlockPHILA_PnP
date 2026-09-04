from ..optim_method import Method
from deepinv.optim.prior import PnP
from deepinv.models import DRUNet
from deepinv.optim.dpir import get_DPIR_params

class DPIR(Method):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prior = PnP(denoiser=DRUNet(pretrained="download", device=self.device))


    def iteration(self, x, y, physics, x_true, filename):


        self.sigma_denoisers, self.taus, self.maxitr = get_DPIR_params(physics.noise_model.sigma, device=self.device)

        x = self.prepare_up(y,physics)
        z = x

        for itr in range(1,self.maxitr+1):
            print('Iteration ', itr)
            x_old = x
            x = self.data_fidelity.prox(z,y,physics, gamma=self.taus[itr-1])
            z = self.prior.prox(x, self.sigma_denoisers[itr-1])


            if self.writer is not None:
                self.save_metrics(x, z, x_old, y, x_true, physics, filename, itr)
        return x, z
        
    