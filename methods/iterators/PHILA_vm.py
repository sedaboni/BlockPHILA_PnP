from ..optim_method import Method
import torch

class PHILA_vm(Method):
    def __init__(self, beta_ls=1e-4, eps_fact=0.5, gamma=1e-4, **kwargs):
        super().__init__(**kwargs)
        
        self.eps_fact = eps_fact
        self.beta_ls = beta_ls
        self.gamma = gamma

        self.metric = True
        
    
    def iteration(self, x, y, physics, x_true, filename):

        self.init_Ds(x)
        
        self.init_bb(x)

        x_old = x

        f_guess = self.data_fidelity(x, y, physics)
        g_guess = self.denoiser.potential(x, sigma=self.sigma_denoiser)
        F_guess = f_guess + self.lamb*g_guess

        for itr in range(1,self.maxitr+1):

            self.update_beta(itr)

            self.rho = 1.0

            F_old = F_guess
            f_x = f_guess

            diff = x-x_old
            norm_diff = diff.norm()**2

            Dg = self.denoiser.potential_grad(x, self.sigma_denoiser)
            self.update_Ds(Dg, itr)
            self.update_bb_step(x, x_old, Dg, itr)

            x_old = x
            
            z = x - (self.tau*self.lamb)*self.Ds*Dg + self.beta_inertial*(diff)
            
            yk = self.data_fidelity.prox_scaled(z,y,physics, xbar=x, gamma=self.tau, D_scaling=1/self.Ds)
            dk = yk - x
            norm_dk = dk.norm()**2

            x_guess = yk
            f_y = self.data_fidelity(yk, y, physics)
            g_y = self.denoiser.potential(yk, sigma=self.sigma_denoiser)
            F_y = f_y + self.lamb*g_y 
            f_guess, g_guess, F_guess = f_y, g_y, F_y
            Delta = ((self.lamb*Dg-(self.beta_inertial/self.tau)*diff/self.Ds)*dk).sum() +((torch.pow(dk,2)/self.Ds).sum())/(2*self.tau) - f_x + f_y.clone()

            LHS = F_y + (0.5*self.gamma*self.rho**2)*norm_dk
            RHS = F_old + (0.5*self.gamma)*norm_diff 
            
            condition_ls =  LHS > RHS + self.beta_ls*self.rho*Delta
            self.counter = 0

            while condition_ls:
                self.rho *= self.eps_fact

                x_guess = x + self.rho*dk

                f_guess = self.data_fidelity(x_guess, y, physics)
                g_guess = self.denoiser.potential(x_guess, sigma=self.sigma_denoiser)
                F_guess = f_guess + self.lamb*g_guess

                LHS = F_guess + (0.5*self.gamma*self.rho**2)*norm_dk
                condition_ls = LHS > RHS + self.beta_ls*self.rho*Delta

                self.counter += 1
                if self.rho < 1e-6:
                    condition_ls = False
                    print(f'Line search failed at iteration {itr} for {filename}')

            if F_y + 0.5*self.gamma*norm_dk <= F_guess + 0.5*self.gamma*norm_dk*(self.rho**2):
                x = yk
                F_guess = F_y
                f_guess = f_y
            else:
                x = x_guess

            if self.writer is not None:
                self.save_metrics(x, z, x_old, y, x_true, physics, filename, itr)
            if self.early_stopping:
                if self.check_convergence(x, x_old, y, physics):
                    print(f'Convergence reached at iteration {itr} for {filename}')
                    break
        z = self.last_step(x)
        return x, z
    
    def update_beta(self, itr):
        self.beta_inertial = (itr-1)/(itr + 2)
    