import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch import Trainer
import os
from torch.utils.data import Dataset
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
from sklearn.preprocessing import QuantileTransformer
from functools import lru_cache
from torch.utils.data import DataLoader
import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.amp import autocast
from torch.optim import Adam
from torchvision import transforms as T, utils
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from scipy.optimize import linear_sum_assignment
from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA
from accelerate import Accelerator
from denoising_diffusion_pytorch.attend import Attend
from denoising_diffusion_pytorch.version import __version__
import matplotlib.pyplot as plt

""
gpu_id=0
""
def set_gpu(gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
if gpu_id >= 0:
    device = "cuda"
    set_gpu(gpu_id)
    print('device available :', torch.cuda.is_available())
    print('device count: ', torch.cuda.device_count())
    print('current device: ',torch.cuda.current_device())
    print('device name: ',torch.cuda.get_device_name())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

config = {
        "input_channels": 1,
        "output_channels": 1,
        "context_image": True,
        "context_channels": 1,
        "num_blocks": [2, 2],
        "hidden_channels": 32,
        "hidden_context_channels": 8,
        "time_embedding_dim": 256,
        "image_size": 128,
        "noise_sampling_coeff": 0.85,
        "denoise_time": 970,
        "activation": "gelu",
        "norm": True,
        "subsample": 100000,
        "save_name": "model_weights.pt",
        "dim_mults": [4, 4],
        "base_dim": 32,
        "timesteps": 1000,
        "pading": "reflect",
        "scaling": "std",
        "optimization": {
            "epochs": 400,
            "lr": 0.01,
            "wd": 0.05,
            "batch_size": 32,
            "scheduler": True
        }
    }

model = Unet(
    channels = 1,
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True,
    self_condition = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = (128, 128),
    timesteps = 1000,    # number of steps
    auto_normalize = True,
    objective = "pred_v",
)
diffusion.is_ddim_sampling = True

trainer = Trainer(
    diffusion,
    '/glade/derecho/scratch/timothyh/data/diffusion_forecasts/processed/lead_0/',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 300,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False,           # whether to calculate fid during training
    max_grad_norm = 1.0,
)
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

trainer.train()
mean = 152.16729736328125
std = 147.735107421875
maxivt = 15.220598
minivt = -1.0300009
data = xr.open_dataset('/glade/derecho/scratch/timothyh/data/diffusion_forecasts/processed/lead_0/train.nc').forecast.sel(time=slice(1,5))
data = (data-mean)/std
data = (data-minivt)/(maxivt-minivt)
X = torch.tensor(data.values).to(device).unsqueeze(1)
sampled_images = diffusion.sample(X, batch_size = 5, return_all_timesteps = False)

obs = sampled_images.detach().cpu().numpy()
image_num=2
plt.imshow(obs[image_num,0,:,:])
plt.colorbar()
plt.savefig("img.png")
