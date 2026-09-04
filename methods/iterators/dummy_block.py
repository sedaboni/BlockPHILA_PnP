from ..block_optim_method import BlockMethod
import torch

class Dummy_block(BlockMethod):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def iteration(self, x, y, physics, x_true, filename):
        z = x
        x_old = z


        torch.cuda.reset_peak_memory_stats(device=self.device)

        base_usage = torch.cuda.max_memory_allocated(device=self.device)
        self.denoiser.potential_grad(x, self.sigma_denoiser, Ui=self.Ui)
        max_usage = torch.cuda.max_memory_allocated(device=self.device)

        usage = max_usage - base_usage

        print(f"gpu used {usage/(1073741824)} memory")


        self.save_metrics(x, z, x_old, y, x_true, physics, filename, 1)

        self.last_step(x)

        return x, z
        
    