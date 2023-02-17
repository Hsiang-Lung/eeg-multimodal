import torch
import functions as aug
import matplotlib.pyplot as plt
from mne import filter
import numpy as np
import os

# script for testing augmentations

def plot_aug_example(x, x_aug, name):
    N = 5500
    fs = 200
    t = torch.arange(0, N) * 1 / fs

    plt.figure(figsize=(15, 4))

    plt.subplot(211)
    plt.plot(x[0])

    plt.subplot(211)
    plt.plot(x_aug[0], alpha=0.7)

    plt.tight_layout()

    plt.savefig(os.getcwd() + '/augmentations/images/' + name + '.png')
    plt.clf()


x = torch.load(os.getcwd() + '/data_splits/multiclass_split/train/data.pt')[0]

for i in range(16):
    x_gauss = aug.gaussian_noise(x, std=0.00002)
    x_timeflip = aug.time_reverse(x)
    x_signflip = aug.sign_flip(x)
    x_ft = aug.ft_surrogate(x, phase_noise_magnitude=0.2, channel_indep=False)[0]
    x_freqshift = aug.frequency_shift(x, delta_freq=0.01, sfreq=200)[0]

    plot_aug_example(x, x_gauss, '0_gauss')
    plot_aug_example(x, x_timeflip, '1_timeflip')
    plot_aug_example(x, x_signflip, '2_signflip')
    plot_aug_example(x, x_ft, '3_ft')
    plot_aug_example(x, x_freqshift, '4_freqsshift')
