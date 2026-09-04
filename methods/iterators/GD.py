from ..optim_method import Method

class GD(Method):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def iteration(self, x, y, physics, x_true, filename):
        z = None
        for itr in range(1,self.maxitr+1):
            x_old = x
            x = x - (self.tau*self.lamb)*self.denoiser.potential_grad(x, self.sigma_denoiser) - self.tau * self.data_fidelity.grad(x,y,physics)
            if self.writer is not None:
                self.save_metrics(x, z, x_old, y, x_true, physics, filename, itr)
            if self.early_stopping:
                if self.check_convergence(x, x_old, y, physics):
                    print(f'Convergence reached at iteration {itr} for {filename}')
                    break
        
        z = self.last_step(x)
        return x, z
        
    