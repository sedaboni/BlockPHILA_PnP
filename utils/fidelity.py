from deepinv.optim.distance import L2Distance
from deepinv.optim.data_fidelity import DataFidelity
import torch
import deepinv

class L2(DataFidelity):

    def __init__(self, sigma=1.0, mode='dual'):
        super().__init__()
        self.d = L2Distance(sigma=sigma)
        self.norm = 1 / (sigma**2)
        self.mode = mode
        self.tau_inexact = 1e3
        self.tau_inexact_scaled = 1e3

    def prox(self, x, y, physics, *args, gamma=1.0, **kwargs):
        return physics.prox_l2(x, y, self.norm * gamma)
    
    def prox_inexact(self, x, y, physics, *args, xbar=None, Uitilde=None, gamma=1.0, **kwargs):
        if Uitilde is None:
            return self.prox(x, y, physics, *args, gamma=gamma, **kwargs)
        else:
            UiCx = xbar-Uitilde(xbar)
            ytilde = y - physics.A(UiCx)
            if self.mode=='dual':
                yk = self.dual(x, ytilde, physics, xbar, UiCx, Uitilde, gamma)
            elif self.mode=='CG':
                yk = self.CG(x, ytilde, physics, UiCx, Uitilde, gamma)
            return yk

    
    def CG(self, x, ytilde, physics, UiCx, Uitilde, gamma):
        physics_restricted = physics*Uitilde
        update = self.prox(x,ytilde,physics_restricted, gamma=gamma)
        yk = UiCx + Uitilde(update)
        return yk
    
    def dual(self, x, ytilde, physics, xold, UiCx, Uitilde, gamma):

        xtemp = Uitilde(x)
        AFx = physics.A(xtemp)
        v = torch.zeros_like(ytilde)

        condition = True
        counter = 0

        norm_xi = (0.5/gamma)*torch.norm(xtemp)**2
        norm_grad = (0.5/gamma)*torch.norm(Uitilde(x-xold))**2
        phi_ui = self.fn(Uitilde(xold),ytilde,physics)

        while condition:
            v = physics.A_adjoint(v)
            v = v - Uitilde(v)
            v = physics.A(v)

            LHS = -ytilde + AFx + gamma*v

            FR = torch.fft.fft2(physics.A_adjoint(LHS))
            x1 = physics.Fh*FR

            FBR = torch.mean(splits(x1, physics.factor), dim=-1, keepdim=False)
            invW = torch.mean(splits(physics.Fh2, physics.factor), dim=-1, keepdim=False)
            invWBR = FBR/(invW+(1/gamma))

            FCBinvWBR = physics.Fhc*invWBR.repeat(1, 1, physics.factor, physics.factor)
            FX = (FR - FCBinvWBR)*gamma

            FX = physics.A(torch.real(torch.fft.ifft2(FX)))

            v = LHS - FX

            update = xtemp - gamma*Uitilde(physics.A_adjoint(v))
            
            psi = -0.5*torch.norm(v)**2 - torch.sum(v*ytilde) - (0.5/gamma)*torch.norm(xtemp-gamma*Uitilde(physics.A_adjoint(v)))**2


            phi = (0.5/gamma)*torch.sum(Uitilde(-2*update*xtemp+update*update)) + self.fn(Uitilde(update),ytilde,physics)


            condition = phi + norm_xi - norm_grad - phi_ui > (2/(2+self.tau_inexact))*(psi + norm_xi - norm_grad - phi_ui)

            counter += 1
            if counter>1000:
                break
        
        print(f'This is the counter {counter}')

        yk = xold + Uitilde(update-xold)
        return yk        
    
    def prox_scaled(self, x, y, physics, *args, xbar=None, Uitilde=None, gamma=1.0, D_scaling=None, **kwargs):

        if torch.std(D_scaling)<1e-3:
            print('Alert: the scaling is a multiple of the identity')

        if self.mode == 'CG':
            if Uitilde is None:
                RHS = physics.A_adjoint(y) + D_scaling * x/gamma

                A_call = lambda s: physics.A_adjoint_A(s) + D_scaling/gamma * s

                y0 = deepinv.optim.utils.conjugate_gradient(A_call, RHS, max_iter=1e3, verbose=True)
            else:
                UiCx = xbar-Uitilde(xbar)
                ytilde = y - physics.A(UiCx)
                physics_restricted = physics*Uitilde
                RHS = physics_restricted.A_adjoint(ytilde) + D_scaling * x/gamma

                A_call = lambda s: physics_restricted.A_adjoint_A(s) + D_scaling/gamma * s

                update = deepinv.optim.utils.conjugate_gradient(A_call, RHS, max_iter=1e3, verbose=True)
                y0 = xbar + Uitilde(update-xbar)

        elif self.mode=='dual':
            if Uitilde is None:
                Uitilde = lambda u: u
            UiCx = xbar-Uitilde(xbar)
            ytilde = y - physics.A(UiCx)

            xtemp = Uitilde(x)
            AFx = physics.A(xtemp)

            norm_xi = (0.5/gamma)*torch.sum(D_scaling*torch.pow(xtemp,2))
            norm_grad = (0.5/gamma)*torch.sum(D_scaling*torch.pow(Uitilde(x-xbar),2))
            phi_ui = self.fn(Uitilde(xbar),ytilde,physics)
            
            v = torch.zeros_like(ytilde)

            condition = True
            counter = 0

            op = lambda u: (u+gamma*physics.A(Uitilde((1/D_scaling)*Uitilde(physics.A_adjoint(u)))))
            RHS = -(- AFx + ytilde)

            r = RHS - op(v)
            p = r
            res_old = torch.sum(r*r)
            
            while condition:

                Ap = op(p)
                alpha = res_old / (torch.sum(p*Ap) + 1e-8)
                v = v + p * alpha
                r = r - Ap * alpha
                res_new = torch.sum(r*r)
                p = r + p * (res_new / (res_old + 1e-8))
                res_old = res_new

                update = xtemp - gamma/D_scaling*Uitilde(physics.A_adjoint(v))
                
                psi = -0.5*torch.norm(v)**2 - torch.sum(v*ytilde) - (0.5/gamma)*torch.sum(D_scaling*torch.pow(update,2))

                phi = (0.5/gamma)*torch.sum(D_scaling*Uitilde(-2*update*xtemp+update*update)) + self.fn(Uitilde(update),ytilde,physics)


                condition = phi + norm_xi - norm_grad - phi_ui > (2/(2+self.tau_inexact_scaled))*(psi + norm_xi - norm_grad - phi_ui)

                counter += 1
                if counter>1000:
                    break
                        
            print(f'This is the counter {counter}')
            y0 = xbar + Uitilde(update-xbar)

        return y0



def splits(a, sf):
    '''split a into sfxsf distinct blocks

    Args:
        a: NxCxWxHx2
        sf: split factor

    Returns:
        b: NxCx(W/sf)x(H/sf)x2x(sf^2)
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
    return b