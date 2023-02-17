import torch
import numpy as np
import pandas as pd
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from yasa import sliding_window
# from sklearn.model_selection import train_test_split

from framework.utils import samplesplit_subsampling, samplesplit_timeframe
from numpy import random
import os

"""
Preprocessing Script for binary classification: healthy vs. patient
"""

DEBUG = False


def undersampling(x, y, seed):
    # first undersamples the patients to have the same amount of samples
    # then undersamples the healthy to have the same amount of the sum of patients
    if DEBUG:
        print("Undersampling...")
        print('Original %s' % Counter(y))
    x_reshaped = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
    rus = RandomUnderSampler(sampling_strategy='not majority', random_state=seed)
    x_res, y_res = rus.fit_resample(x_reshaped, y)

    if DEBUG: print('Resampled dataset shape %s' % Counter(y_res))
    y_res[y_res != 0] = 1
    rus = RandomUnderSampler(sampling_strategy='not minority', random_state=seed)
    x_res, y_res = rus.fit_resample(x_res, y_res)
    x_res = np.reshape(x_res, (x_res.shape[0], x.shape[1], x.shape[2]))
    if DEBUG: print('Resampled dataset shape %s' % Counter(y_res))
    return torch.from_numpy(x_res), y_res


def oversampling(x, y, seed):
    # first oversamples the patients to have the same amount of samples
    # then oversamples the healthy to have the same amount of the sum of patients
    if DEBUG:
        print("Oversampling...")
        print('Original %s' % Counter(y))
    x_reshaped = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
    rus = RandomOverSampler(sampling_strategy='not majority', random_state=seed)
    x_res, y_res = rus.fit_resample(x_reshaped, y)

    if DEBUG: print('Resampled dataset shape %s' % Counter(y_res))
    y_res[y_res != 0] = 1
    rus = RandomOverSampler(sampling_strategy='not majority', random_state=seed)
    x_res, y_res = rus.fit_resample(x_res, y_res)
    x_res = np.reshape(x_res, (x_res.shape[0], x.shape[1], x.shape[2]))
    if DEBUG: print('Resampled dataset shape %s' % Counter(y_res))
    return torch.from_numpy(x_res), y_res


def preprocessing_binary(x, y, dest_path, configs, random_seed):
    # applies (over/under-)sampling and sample splitting based on the config and saves the preprocessced tensors
    if DEBUG: print('Class %s' % Counter(y))
    x_train, y_train, x_test, y_test = train_test_split(x, y)

    sampling = configs['sampling']
    if sampling == 'undersampling':
        x_train, y_train = undersampling(x_train, y_train, random_seed)
    elif sampling == 'oversampling':
        x_train, y_train = oversampling(x_train, y_train, random_seed)

    sample_split = configs['sample_split']
    split_count = 1
    freq = 200
    if 'subsampling' in sample_split:
        split_count = configs['subsampling_count']
        x_train, y_train, _ = samplesplit_subsampling(x_train, y_train, split_count)
        x_test, y_test, _ = samplesplit_subsampling(x_test, y_test, split_count)
        freq = freq / split_count
    if 'segments' in sample_split:
        x_train, y_train, _, split_c = samplesplit_timeframe(x_train, y_train,
                                                             freq=freq,
                                                             window=configs['segment_window'],
                                                             step=configs['segment_step'])
        x_test, y_test, _, _ = samplesplit_timeframe(x_test, y_test,
                                                     freq=freq,
                                                     window=configs['segment_window'],
                                                     step=configs['segment_step'])
        split_count *= split_c
    if DEBUG:
        print('TRAIN/TEST X SHAPES:' + str((x_train.shape, x_test.shape)))
        print('TRAIN/TEST Y SHAPES:' + str((y_train.shape, y_test.shape)))

    torch.save(torch.from_numpy(x_train).float(), os.getcwd() + dest_path + 'train/data.pt')
    torch.save(torch.from_numpy(y_train), os.getcwd() + dest_path + 'train/labels.pt')
    torch.save(torch.from_numpy(x_test).float(), os.getcwd() + dest_path + 'test/data.pt')
    torch.save(torch.from_numpy(y_test), os.getcwd() + dest_path + 'test/labels.pt')

    print('Class (train) %s' % Counter(y_train))
    print('Class (test) %s' % Counter(y_test))

    return split_count


def train_test_split(x, y):
    y0_test_idx = np.random.choice(np.where(y == 0)[0], 9)
    y1_test_idx = np.random.choice(np.where(y == 1)[0], 3)
    y2_test_idx = np.random.choice(np.where(y == 2)[0], 3)
    y3_test_idx = np.random.choice(np.where(y == 3)[0], 3)
    y_test_idx = np.concatenate((y0_test_idx, y1_test_idx, y2_test_idx, y3_test_idx))

    x_test = x[y_test_idx]
    y_test = y[y_test_idx]
    y_test[y_test != 0] = 1

    x_train = np.delete(x, y_test_idx, axis=0)
    y_train = np.delete(y, y_test_idx)
    y_train[y_train != 0] = 1
    return x_train, y_train, x_test, y_test

# sex = table[:, 1]
# spliter = np.concatenate((sex[:, np.newaxis], y[:, np.newaxis]), axis=1)
# x_train, x_test, y_train, y_test = TTS(x, spliter, random_state=123, stratify=spliter)
# table_train, table_test, _, _ = TTS(table, spliter, random_state=123, stratify=spliter)
