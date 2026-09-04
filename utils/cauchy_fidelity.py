import torch
from deepinv.optim.data_fidelity import DataFidelity
from deepinv.physics import NoiseModel
import deepinv as dinv

class CauchyFidelity(DataFidelity):
    def __init__(self, gamma=0.01):
        super().__init__()
        self.gamma = gamma

    def forward(self, x, y, physics, **kwargs):
        Ax = physics.A(x)
        residual = (Ax - y)
        return 0.5 * torch.sum(torch.log(self.gamma**2+residual**2))

    def grad(self, x, y, physics, **kwargs):
        Ax = physics.A(x)
        diff = Ax - y
        grad_u = diff / (self.gamma**2 + diff**2)
        return physics.A_adjoint(grad_u)

class CauchyNoise(NoiseModel):
    def __init__(
        self,
        gamma: float | torch.Tensor = 0.01,
        clip: bool = True,
        rng: torch.Generator | None = None,
    ):
        device = dinv.physics.noise._infer_device([gamma, rng])
        super().__init__(rng=rng)
 
        self.clip = clip

        gamma = self._float_to_tensor(gamma)
        gamma = gamma.to(device)
        self.register_buffer("gamma", gamma)

    def forward(self, x, gamma=None, seed: int = None, **kwargs):
        self.update_parameters(gamma=gamma, **kwargs)
        self.rng_manual_seed(seed)
        self.to(x.device)
        gamma = self.gamma[(...,) + (None,) * (x.dim() - 1)]

        cauchy_dist = torch.distributions.Cauchy(loc=0.0, scale=self.gamma)
        noise = cauchy_dist.rsample(x.shape).to(x.device)
        y = x + noise

        if self.clip:
            y =  torch.clamp(y, 0.0, 1.0)
        return y