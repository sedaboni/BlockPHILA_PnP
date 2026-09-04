import argparse
import os
import random
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from methods.utils import get_optimizer
from utils.generate_dataset import MyDataset
from utils.utils import save_image
from utils.cauchy_fidelity import CauchyNoise
import deepinv as dinv
import numpy as np
import torch
import matplotlib.pyplot as plt

# Reproducibility: disable cuDNN autotuning and keep it deterministic.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed for deterministic CUDA matmuls
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def set_seed(seed):
    """Seed every RNG that affects the pipeline (Python, NumPy, PyTorch, CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # warn_only=True: a non-deterministic op degrades to a warning instead of raising.
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(device_arg):
    """Pick the compute device: explicit --device wins, else the freest GPU, else CPU."""
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return dinv.utils.get_freer_gpu()
    return "cpu"

def main():

    parser = argparse.ArgumentParser(description="Another toolbox to compare stuff")
    parser.add_argument('--opt', type=str, help='Optimizer Name', default='GD_block')
    parser.add_argument('--nblocks', type=int, help='Number of blocks', default=1)
    parser.add_argument('--maxit', type=int, help='Max iter', default=1000)
    parser.add_argument('--sf', type=int, help='Scale factor', default=1)
    parser.add_argument('--dataset_name', type=str, default='small')
    parser.add_argument('--fidelity', type=str, default='cauchy')
    parser.add_argument('--crit_conv', type=str, default='cost') #'cost''time'
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--show_img', action='store_true')
    parser.add_argument('--save_img', action='store_true')
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--gamma', type=float, default=1e-2)
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--lamb', type=float, default=0.075)
    parser.add_argument('--thres_conv', type=float, default=1e-5)
    parser.add_argument('--BB', type=str, default=None)
    parser.add_argument('--prox', type=str, default='CG')
    parser.add_argument('--den_type', type=str, default='drunet')
    parser.add_argument('--seed', type=int, default=142, help='Global RNG seed')
    parser.add_argument('--device', type=str, default=None, help="Compute device, e.g. 'cuda:0' or 'cpu'. Default: freest GPU, else CPU.")
    parser.set_defaults(grayscale=False, show_img=False, save_img=True, early_stopping=True)

    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    print(f'Using device: {device} | seed: {args.seed}')

    dataset = MyDataset(path=args.dataset_name, device=device, grayscale=args.grayscale)
    img_size = dataset.get_img_size()

    gamma = args.gamma
    noise_model = CauchyNoise(gamma=gamma)

    filter = dinv.physics.blur.gaussian_blur(sigma=(1.6, 1.6))
    
    physics = dinv.physics.Downsampling(factor=args.sf, filter=filter, img_size=img_size,\
                                noise_model=noise_model, device=device)
    dataset.set_physics(physics)

    if args.sf == 1:
        lamb = 1e2
        sigma_k_denoiser = 1.
    else:
        lamb = args.lamb
        sigma_k_denoiser = 1.


    tau = 1/(gamma**(-2)+lamb)
    sigma_denoiser = (sigma_k_denoiser * 25)/255 # denoiser expects a noise level tensor
    print('Sigma_denoiser: ', sigma_denoiser)

    log_dir = f'results_cauchy/sf_{args.sf}_noise_{args.gamma}_corr/nb_{args.nblocks}/{args.opt}_{args.fidelity}_{args.comment}_{args.prox}_{args.lamb}_{args.BB}'
    
    writer = SummaryWriter(log_dir=log_dir)
    
    optimizer = get_optimizer(args.opt)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for el in dataloader:

        im_true, im_degraded, filename = el
        
        method = optimizer(lamb=lamb, tau=tau, sigma_denoiser=sigma_denoiser, img_size=img_size, nblocks=args.nblocks, writer=writer, device=device, maxitr=args.maxit, fidelity=args.fidelity, BB=args.BB, prox_mode=args.prox, crit_conv =args.crit_conv, thres_conv=args.thres_conv, early_stopping=args.early_stopping, den_type=args.den_type, gamma_cauchy=args.gamma)

        im_rec, z = method(im_degraded, physics, im_true, filename[0])

        if args.show_img:
            dinv.utils.plot([im_true[0], im_degraded[0], im_rec[0], z[0]], titles=['True', 'Degraded', 'Recon', 'z'], figsize=(10, 5))
        if args.save_img:
            save_image(im_rec, filename[0], log_dir, flag='x')
            save_image(z, filename[0], log_dir, flag='z')
            save_image(im_degraded, filename[0], log_dir, flag='deg')

if __name__ == "__main__":
    main()