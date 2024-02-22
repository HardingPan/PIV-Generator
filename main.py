import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

import torch

from config import config
from generator import Generator
from plot import plot_field

def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    return True

def flowtest(cfg):
    pig = Generator(cfg)
    img1, img2, ut, vt = pig.compute()
    
    # show the velocity field
    x, y = np.meshgrid(np.arange(ut.shape[0]), np.arange(ut.shape[1]), indexing="ij")
    amp = np.sqrt(ut**2+vt**2)
    fig = plot_field(x,y,ut,vt,bkg=amp,cmap=None,figsize=(8,3.2))

if __name__ == '__main__':
    """
    argument init
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default="uniform_flow", \
                        choices=["uniform_flow","sine_flow","lamboseen_flow","cellular_flow"], \
                        help="set the type of particle image")
    parser.add_argument('--path', default="/media/newdisk/generated_piv", \
                        help="save the generated PIV in this path")
    parser.add_argument('--output', default="npy", \
                        choices=["npy","tif","png"], \
                        help="set the file type of the output")
    parser.add_argument('--num', default=10, \
                        help="set The amount of data generated")
    parser.add_argument('--seed', default=1234, help="set the random seeds")
    parser.add_argument('--value', type=list, nargs='+', help="set the flow field value")

    args = parser.parse_args()
    
    set_seed(args.seed)
    
    cfg = config()
    cfg.flow = args.type
    if args.value:
        if args.type == "uniform_flow":
            assert len(args.value) > 1
            cfg.uniform_flow['c_x'] = float(''.join(args.value[0]))
            cfg.uniform_flow['c_y'] = float(''.join(args.value[0]))
        elif args.type == "sine_flow":
            assert len(args.value) > 2
            cfg.sine_flow['a'] = int(''.join(args.value[0]))
            cfg.sine_flow['px'] = int(''.join(args.value[1]))
            cfg.sine_flow['vmax'] = float(''.join(args.value[2]))
        elif args.type == "lamboseen_flow":
            cfg.lamboseen_flow['gamma'] = float(''.join(args.value[0]))
        elif args.type == "cellular_flow":
            cfg.cellular_flow['vmax'] = float(''.join(args.value[0]))
    
    """
    Generate
    """
    pig = Generator(cfg)
    img1, img2, ut, vt = pig.compute()
    
    # a pair of PIV
    plt.figure()
    plt.subplot(121); plt.imshow(img1, cmap="gray"); plt.axis("off")
    plt.subplot(122); plt.imshow(img1, cmap="gray"); plt.axis("off")
    plt.savefig("particle_image.png")

    # the velocity field
    x, y = np.meshgrid(np.arange(ut.shape[0]), np.arange(ut.shape[1]), indexing="ij")
    amp = np.sqrt(ut**2 + vt**2)
    fig = plot_field(x, y, ut, vt, bkg=amp, cmap=None, figsize=(8, 3.2))
    plt.savefig("velocity_field.png")
    
    print("over")
    