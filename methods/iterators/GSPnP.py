from ..optim_method import Method
import torch

class GSPnP(Method):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.eta_backtraking = 0.9
        self.gamma_backtraking = 0.1
        self.backtracking_check = True
    
    def iteration(self, x, y, physics, x_true, filename):
        self.backtracking_check = True
        z = None
        F_new = None
        self.rho = 1.
        for itr in range(1,self.maxitr+1):
            F_old = F_new
            x_old = x
            z = x - (self.rho*self.tau*self.lamb)*self.denoiser.potential_grad(x, self.sigma_denoiser)
            x = self.data_fidelity.prox(z,y,physics, gamma=self.tau*self.rho)
            # Backtracking
            f_new = self.data_fidelity(x, y, physics)
            g_new = self.denoiser.potential(x, sigma=self.sigma_denoiser)
            F_new = f_new + self.lamb*g_new
            if itr>1:
                diff_x = torch.norm(x-x_old)**2
                diff_F = F_old - F_new
                if diff_F < (self.gamma_backtraking/(self.rho*self.tau))*diff_x:
                    self.rho *= self.eta_backtraking
                    x = x_old
                    F_new = F_old
                    self.backtracking_check = False
                else:
                    self.backtracking_check = True

            if self.writer is not None:
                self.save_metrics(x, z, x_old, y, x_true, physics, filename, itr)
            if self.early_stopping and (self.backtracking_check or self.crit_conv=='time'):
                if self.check_convergence(x, x_old, y, physics):
                    print(f'Convergence reached at iteration {itr} for {filename}')
                    break
        
        z = self.last_step(x)
        return x, z
        
    