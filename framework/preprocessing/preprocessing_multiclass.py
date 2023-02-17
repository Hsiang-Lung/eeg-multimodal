import os
from collections import Counter

import torch
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import pandas as pd

from framework.utils import samplesplit_subsampling, samplesplit_timeframe

"""
Preprocessing Script for multiclass classification: chronic bp vs. fibromyalgia vs various
Divides Data into train val test using hardcoded values (for now)
"""

DEBUG = False


# UNDERSAMPLING (for train)
def undersampling(x, y, table, seed):
    if DEBUG:
        print("Undersampling...")
        print('Original %s' % Counter(y))
    x_reshaped = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
    rus = RandomUnderSampler(sampling_strategy='not minority', random_state=seed)
    x_res, y_res = rus.fit_resample(x_reshaped, y)
    table, _ = rus.fit_resample(table, y)
    x_res = np.reshape(x_res, (x_res.shape[0], x.shape[1], x.shape[2]))
    if DEBUG: print('Resampled dataset shape %s' % Counter(y_res))
    return torch.from_numpy(x_res), y_res, table


# OVERSAMPLING (for train)
def oversampling(x, y, table, seed):
    if DEBUG:
        print("Oversampling...")
        print('Original %s' % Counter(y))
    x_reshaped = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
    rus = RandomOverSampler(sampling_strategy='not majority', random_state=seed)
    x_res, y_res = rus.fit_resample(x_reshaped, y)
    table, y_res2 = rus.fit_resample(table, y)
    x_res = np.reshape(x_res, (x_res.shape[0], x.shape[1], x.shape[2]))
    if DEBUG: print('Resampled dataset shape %s' % Counter(y_res))
    return torch.from_numpy(x_res), y_res, table


def normalize_table(table):
    for i in range(len(table[0])):
        mean = np.nanmean(table[:, i])
        std = np.nanstd(table[:, i])
        table[:, i] = (table[:, i] - mean) / std
    return table


def preprocessing_multiclass(x, y, table, original_table, dest_path, configs, random_seed, train_idx=None,
                             test_idx=None):
    # applies (over/under-)sampling and sample splitting based on the config and saves the preprocessced tensors
    # removes healthy samples
    if DEBUG: print('Class %s' % Counter(y))

    if train_idx is not None and test_idx is not None:
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        table_train, table_test = table[train_idx], original_table[test_idx]
    else:
        patient_indices = [y != 0]
        x = x[tuple(patient_indices)]
        y = y[tuple(patient_indices)] - 1
        table = table[tuple(patient_indices)]
        table = normalize_table(table)
        original_table = normalize_table(original_table)
        x_train, x_test, y_train, y_test, train_idx, test_idx = train_test_split(x, y, np.arange(len(y)),
                                                                                 random_state=random_seed, stratify=y)
        table_train, table_test = table[train_idx], original_table[test_idx]

    sampling = configs['sampling']
    if sampling == 'undersampling':
        x_train, y_train, table_train = undersampling(x_train, y_train, table_train, random_seed)
    elif sampling == 'oversampling':
        x_train, y_train, table_train = oversampling(x_train, y_train, table_train, random_seed)

    sample_split = configs['sample_split']
    split_count = 1
    freq = 200
    if 'subsampling' in sample_split:
        split_count = int(configs['subsampling_count'])
        x_train, y_train, table_train = samplesplit_subsampling(x_train, y_train, split_count, table_train)
        x_test, y_test, table_test = samplesplit_subsampling(x_test, y_test, split_count, table_test)
        freq = freq / split_count
    if 'segments' in sample_split:
        x_train, y_train, table_train, split_c = samplesplit_timeframe(x_train, y_train, table_train,
                                                                       freq=freq,
                                                                       window=configs['segment_window'],
                                                                       step=configs['segment_step'])
        x_test, y_test, table_test, _ = samplesplit_timeframe(x_test, y_test, table_test,
                                                              freq=freq,
                                                              window=configs['segment_window'],
                                                              step=configs['segment_step'])
        split_count *= split_c
    if DEBUG:
        print('TRAIN/TEST X SHAPES:' + str((x_train.shape, x_test.shape)))
        print('TRAIN/TEST Y SHAPES:' + str((y_train.shape, y_test.shape)))

    torch.save(torch.from_numpy(x_train).float(), os.getcwd() + dest_path + 'train/data.pt')
    torch.save(torch.from_numpy(y_train), os.getcwd() + dest_path + 'train/labels.pt')
    torch.save(torch.from_numpy(table_train).float(), os.getcwd() + dest_path + 'train/table.pt')

    torch.save(torch.from_numpy(x_test).float(), os.getcwd() + dest_path + 'test/data.pt')
    torch.save(torch.from_numpy(y_test), os.getcwd() + dest_path + 'test/labels.pt')
    torch.save(torch.from_numpy(table_test).float(), os.getcwd() + dest_path + 'test/table.pt')

    print('Class (train) %s' % Counter(y_train))
    print('Class (test) %s' % Counter(y_test))

    return split_count
