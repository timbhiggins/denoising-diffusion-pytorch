import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import os
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
from sklearn.preprocessing import QuantileTransformer
from functools import lru_cache
from torch.utils.data import Dataset, DataLoader
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
from denoising_diffusion_pytorch.Train import Trainer_ARs, GaussianDiffusion_AR
from denoising_diffusion_pytorch.Train import transform_sampled_images

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

# small helper modules
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

model = Unet(
    channels = 1,
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True,
    self_condition = True
)

diffusion = GaussianDiffusion_AR(
    model,
    image_size = (128, 128),
    timesteps = 1000,    # number of steps
    auto_normalize = True,
    objective = "pred_v",
)
diffusion.is_ddim_sampling = True

trainer = Trainer_ARs(
    diffusion,
    '/glade/derecho/scratch/timothyh/data/diffusion_forecasts/processed/',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 10000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False,           # whether to calculate fid during training
    max_grad_norm = 1.0,
)
trainer.train()
sampled_images = diffusion.sample(batch_size = 50, return_all_timesteps = False)
sampled_images_t = transform_sampled_images(sampled_images, trainer)

obs = sampled_images.detach().cpu().numpy()
image_num=19
plt.imshow(obs[image_num,0,:,:])
plt.colorbar()
plt.savefig("img.png")
