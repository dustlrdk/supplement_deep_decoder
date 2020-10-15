from __future__ import print_function
import matplotlib.pyplot as plt
# %matplotlib notebook

import os

import warnings

from include import *
from PIL import Image
import PIL

import numpy as np
import torch
import torch.optim
from torch.autograd import Variable

GPU = True
if GPU == True:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
    print("num GPUs", torch.cuda.device_count())
else:
    dtype = torch.FloatTensor

# %% md
## Load image
# %%
import sys
path = './test_data/CSet9'
img_name = sys.argv[1]
# img_name = "phantom256"

img_path = path + img_name + ".png"
img_pil = Image.open(img_path)
img_np = pil_to_np(img_pil)
img_clean_var = np_to_var(img_np).type(dtype)


# %% md
## Generate noisy image
# %%
def get_noisy_img(sig=25, noise_same=False):
    print("[*] sigma : %d" % sig)
    sigma = sig / 255.
    if noise_same:  # add the same noise in each channel
        noise = np.random.normal(scale=sigma, size=img_np.shape[1:])
        noise = np.array([noise] * img_np.shape[0])
    else:  # add independent noise in each channel
        noise = np.random.normal(scale=sigma, size=img_np.shape)

    img_noisy_np = np.clip(img_np + noise, 0, 1).astype(np.float32)
    img_noisy_var = np_to_var(img_noisy_np).type(dtype)
    return img_noisy_np, img_noisy_var


img_noisy_np, img_noisy_var = get_noisy_img()
output_depth = img_np.shape[0]
print("Image size: ", img_np.shape)


# %% md
## Denoise noisy image
# %%
def denoise(img_noisy_var, k=128, numit=1900, rn=0.0, find_best=True, upsample_first=True):
    num_channels = [k] * 5
    net = decodernw(output_depth, num_channels_up=num_channels, upsample_first=upsample_first).type(dtype)
    mse_n, mse_t, ni, net = fit(num_channels=num_channels,
                                reg_noise_std=rn,
                                num_iter=numit,
                                img_noisy_var=img_noisy_var,
                                net=net,
                                img_clean_var=img_clean_var,
                                find_best=find_best
                                )
    out_img_np = net(ni.type(dtype)).data.cpu().numpy()[0]
    return out_img_np, mse_t


# %%
def myimgshow(plt, img):
    plt.imshow(np.clip(img.transpose(1, 2, 0), 0, 1))


# def plot_results(out_img_np, img_np, img_noisy_np):
#     fig = plt.figure(figsize=(15, 15))  # create a 5 x 5 figure
#
#     ax1 = fig.add_subplot(131)
#     myimgshow(ax1, img_np)
#     ax1.set_title('Original image')
#     ax1.axis('off')
#
#     ax2 = fig.add_subplot(132)
#     myimgshow(ax2, img_noisy_np)
#     ax2.set_title("Noisy observation, PSNR: %.2f" % psnr(img_np, img_noisy_np))
#     ax2.axis('off')
#
#     ax3 = fig.add_subplot(133)
#     myimgshow(ax3, out_img_np)
#     ax3.set_title("Deep-Decoder denoised image, SNR: %.2f" % psnr(img_np, out_img_np))
#     ax3.axis('off')
#
#     plt.show()


img_noisy_np, img_noisy_var = get_noisy_img(sig=25, noise_same=False)
out_img_np, mse_t = denoise(img_noisy_var, k=128, numit=1900, rn=0.0)
np.save("%s_DD.npy"% img_name)


# # plot_results(out_img_np, img_np, img_noisy_np)
# # %%
# img_noisy_np, img_noisy_var = get_noisy_img(sig=25, noise_same=False)
# out_img_np, mse_t = denoise(img_noisy_var, k=128, numit=10000, rn=0.015, upsample_first=False)
#
# # plot_results(out_img_np, img_np, img_noisy_np)
# %% md
# ## Choice of number of layers
#
# The
# number
# of
# layers $k$ is a
# hyperparameter
# that
# enables
# trading
# off
# amount
# of
# noise
# that is removed
# versus
# the
# representation
# error
# by
# the
# model.
# Smaller $k$ remove
# more
# of
# the
# noise, but
# also
# increase
# the
# error
# of
# approximating
# an
# image
# with the deep decoder.The optimal choise of $k$ depends on the noise level.To illustrate this consider the experiment below.
#
# The
# following
# series
# of
# plots
# show
# the
# PSNR
# of
# the
# output
# of
# the
# deep
# decoder
# during
# training as a
# function
# of
# iteration
# number.The
# blue
# curves
# correspond
# to
# k = 32, the
# orange
# corresponds
# to
# k = 64, and the
# green
# corresponds
# to
# k = 128.


# %%
# def best_k(ks, sig=70, numit=400, noise_same=False, find_best=True):
#     img_noisy_np, img_noisy_var = get_noisy_img(sig=sig, noise_same=noise_same)
#     print("Noisy observation, PSNR: %.2f" % psnr(img_np, img_noisy_np))
#     mses = []
#     psnrs = []
#     for k in ks:
#         out_img_np, mse_t = denoise(img_noisy_var, k=k, numit=numit, rn=0.0)
#         psnrs += [psnr(img_np, out_img_np)]
#         mses += [mse_t]
#     plt.yscale('log')
#     plt.xscale('log')
#     for mse_t in mses:
#         plt.plot(mse_t)
#     plt.show()
#     print(psnrs)


# # %% md
# ### If we regularize with the model alone, and run close to convergence, $64$ performs best:
# # %%
# ks = [32, 64, 128]
# best_k(ks, sig=25, numit=10000, noise_same=True)
# # %% md
# ### If we additionally stop early to regularize, then $k=128$ performs best:
# # %%
# ks = [32, 64, 128]
# best_k(ks, sig=25, numit=1900, noise_same=True)
# # %% md
# ### More noise requires more regularization, either by using a smaller $k$:
# # %%
# ks = [32, 64, 128]
# best_k(ks, sig=60, numit=5000, noise_same=True)
# # %% md
# ### ... or by stopping even earlier:
# # %%
# ks = [32, 64, 128]
# best_k(ks, sig=60, numit=800, noise_same=True)