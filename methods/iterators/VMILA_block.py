from ..block_optim_method import BlockMethod
import torch
import deepinv as dinv

class VMILA_block(BlockMethod):
    def __init__(self, beta_ls=1e-4, eps_fact=0.5, **kwargs):
        super().__init__(**kwargs)
        
        self.eps_fact = eps_fact
        self.beta_ls = beta_ls

        
    
    def iteration(self, x, y, physics, x_true, filename):
        
        self.init_bb(x)
        x_old = torch.zeros_like(x)

        f_guess = self.data_fidelity(x, y, physics)
        g_guess = self.denoiser.potential(x, sigma=self.sigma_denoiser)
        F_guess = f_guess + self.lamb*g_guess


        for itr in range(1,self.maxitr+1):

            self.rho = 1.0

            F_old = F_guess
            f_x = f_guess 

            Dg = self.denoiser.potential_grad(x, self.sigma_denoiser, Ui=self.Ui)
            self.update_bb_step(x, x_old, Dg, itr)
            x_old = x_old - self.Uitilde(x_old - x)

            z = x - (self.tau[self.block_counter]*self.lamb)*Dg

            
            yk = self.data_fidelity.prox_inexact(z,y, physics, xbar=x, Uitilde=self.Uitilde, gamma=self.tau[self.block_counter])
            dk = yk - x

            x_guess = yk
            f_y = self.data_fidelity(yk, y, physics)
            g_y = self.denoiser.potential(yk, sigma=self.sigma_denoiser)
            F_y = f_y + self.lamb*g_y 
            f_guess, g_guess, F_guess = f_y, g_y, F_y
            Delta = self.lamb*(Dg*dk).sum() + (dk.norm()**2)/(2*self.tau[self.block_counter]) - f_x + f_y
            
            condition_ls = F_y > F_old + self.beta_ls*self.rho*Delta
            self.counter = 0

            while condition_ls:
                self.rho *= self.eps_fact

                x_guess = x + self.rho*dk

                f_guess = self.data_fidelity(x_guess, y, physics)
                g_guess = self.denoiser.potential(x_guess, sigma=self.sigma_denoiser)
                F_guess = f_guess + self.lamb*g_guess

                condition_ls = F_guess > F_old + self.beta_ls*self.rho*Delta

                self.counter += 1
                if self.rho < 1e-6:
                    condition_ls = False
                    print(f'Line search failed at iteration {itr} for {filename}')
            
            if F_y < F_guess:
                x = yk
                F_guess = F_y
                f_guess = f_y
            else:
                x = x_guess
            


            self.update_block()

            if self.writer is not None:
                self.save_metrics(x, z, x_old, y, x_true, physics, filename, itr)
            if self.early_stopping:
                if self.check_convergence(x, x_old, y, physics):
                    print(f'Convergence reached at iteration {itr} for {filename}')
                    break
        z = self.last_step(x)
        return x, z
        
    