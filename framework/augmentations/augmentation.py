from typing import Callable
import torch
import numpy as np
from copy import deepcopy
import framework.augmentations.functions as aug
from mne import filter

"""
Callable Classes for each Augmentation to be used in the __getitem__ method of the EEG Dataset
"""


class GaussianNoise(Callable):
    def __init__(self, mu, sigma, channel_prob):
        self.mu = mu
        self.sigma = sigma
        self.channel_prob = channel_prob

    def __call__(self, signal):
        noisy_signal = deepcopy(signal)
        N = len(noisy_signal[0])
        noise = torch.normal(mean=torch.full((1, N), self.mu, dtype=torch.float16),
                             std=torch.full((1, N), self.sigma, dtype=torch.float16))

        for idx, channel in enumerate(noisy_signal):
            if self.channel_prob > np.random.uniform():
                noisy_signal[idx] = noisy_signal[idx] + noise[0]

        return noisy_signal


class SignFlip(Callable):
    def __init__(self, channel_prob):
        self.channel_prob = channel_prob

    def __call__(self, signal):
        if self.channel_prob > np.random.uniform():
            return -signal
        else:
            return signal


class TimeFlip(Callable):
    def __init__(self, channel_prob):
        self.channel_prob = channel_prob

    def __call__(self, signal):
        if self.channel_prob > np.random.uniform():
            return torch.flip(signal, [1])
        else:
            return signal


class RandomRescale(Callable):
    def __init__(self, mu, sigma, channel_prob):
        self.mu = mu
        self.sigma = sigma
        self.channel_prob = channel_prob

    def __call__(self, signal):
        if self.channel_prob > np.random.uniform():
            rescaling = torch.normal(mean=torch.Tensor([self.mu]), std=torch.Tensor([self.sigma]))
            return signal * rescaling
        else:
            return signal


class FourierTransformSurrogate(Callable):
    def __init__(self, phase_noise_magnitude, prob, channel_indep=False):
        self.phase_noise_magnitude = phase_noise_magnitude
        self.channel_indep = channel_indep
        self.prob = prob

    def __call__(self, signal):
        if self.prob > np.random.uniform():
            return aug.ft_surrogate(signal, self.phase_noise_magnitude, self.channel_indep)[0]
        else:
            return signal


class FrequencyShift(Callable):
    def __init__(self, delta_freq, prob, sfreq=200):
        self.delta_freq = delta_freq
        self.sfreq = sfreq
        self.prob = prob

    def __call__(self, signal):
        if self.prob > np.random.uniform():
            return aug.frequency_shift(signal, self.delta_freq, self.sfreq)[0]
        else:
            return signal


class ChannelDropout(Callable):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, signal):
        return aug.channels_dropout(signal, self.prob)


class ColumnDropout(Callable):
    def __init__(self, prob, indices):
        self.indices = indices
        self.prob = prob

    def __call__(self, table):
        return aug.column_dropout(table, self.prob, self.indices)
