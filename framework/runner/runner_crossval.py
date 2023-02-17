import json
import os
import sys

import torch
import numpy as np 
import random

from sklearn.model_selection import StratifiedKFold

sys.path.append(os.getcwd())
os.environ["OUTDATED_IGNORE"] = "1"
import framework.preprocessing.preprocessing as pp
import framework.training as training
from framework.utils import get_augmentations, set_seed, calc_splitcount
from framework.preprocessing.preprocessing_multiclass import normalize_table
from framework.utils import normalization, bandpass_filter, notch_filter, plot_roc_crossval

import warnings
import torch

warnings.filterwarnings('ignore')

'''
Runs training based on the contents of configs_filter.json

'''
config_file = 'config.json'
with open(os.getcwd() + '/framework/runner/' + config_file) as f:
    configs = json.load(f)

for run in configs['runs']:
    set_seed(run['random_seed'])
    # load tensors  and table
    src_path = run['preprocessing']['src_path']

    x_dinh = torch.load(src_path + 'data_Dinh_fs200.pt')
    y_dinh = torch.load(src_path + 'labels_Dinh_fs200.pt')

    x_heitmann = torch.load(src_path + 'data_Heitmann_fs200.pt')
    y_heitmann = torch.load(src_path + 'labels_Heitmann_fs200.pt')

    x = torch.cat((x_dinh, x_heitmann), dim=0).numpy()

    y = torch.cat((y_dinh, y_heitmann), dim=0).numpy()

    # Filter (notch or bandpass filter)
    filter = run['preprocessing']['filter']
    if filter == 'notch':
        x = notch_filter(x, 200, 50)
    elif filter == 'bandpass':
        x = bandpass_filter(x, 200, 1, 45)

    # Normalize (channel, sample or dataset wise)
    x = normalization(x, run['preprocessing']['normalization'])
    patient_indices = [y != 0]
    x = x[tuple(patient_indices)]
    y = y[tuple(patient_indices)] - 1
    filled_table, original_table = pp.get_tabular_data(src_path)
    filled_table = filled_table[tuple(patient_indices)]
    original_table = original_table[tuple(patient_indices)]
    filled_table = normalize_table(filled_table)
    original_table = normalize_table(original_table)
    y_list, y_pred_list = [], []

    splits = 4
    skf = StratifiedKFold(n_splits=splits, random_state=run['random_seed'], shuffle=True)
    for i, (train_index, test_index) in enumerate(skf.split(x, y)):
        print("FOLD: ", i)
        split_count_true = pp.preprocessing_crossval(x=x, y=y, table=filled_table, original_table=original_table,
                                                configs=run['preprocessing'], random_seed=run['random_seed'],
                                                train_idx=train_index, test_idx=test_index)        
        split_count = calc_splitcount(run['preprocessing'])
        assert split_count_true==split_count

        print("DONE PREPROCESSING - START TRAINING")
        test_y, test_y_pred = training.run_training(
            data_path=run['preprocessing']['dest_path'],
            transform_train=get_augmentations(run['augmentations']),
            device='cuda:3',
            binary=run['preprocessing']['binary'],
            run_id=run['run_id'],
            configs=run['model'],
            split_count=split_count,
            fold='/fold' + str(i),
            folds=splits)

        y_list.append(test_y)
        y_pred_list.append(test_y_pred)

    plot_roc_crossval(y_list,y_pred_list, run['run_id'])
