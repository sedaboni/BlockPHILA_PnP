from ..optim_method import Method
import torch
import matplotlib.pyplot as plt

class Dummy(Method):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def iteration(self, x, y, physics, x_true, filename):
        z = x
        x_old = z

        self.save_metrics(x, z, x_old, y, x_true, physics, filename, 0)
        return x, z
        
    