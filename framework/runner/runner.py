import json
import os
import sys
import numpy as np
import random

sys.path.append(os.getcwd())
os.environ["OUTDATED_IGNORE"] = "1"
import framework.preprocessing.preprocessing as pp
import framework.training as training
from framework.utils import get_augmentations, set_seed, calc_splitcount, createFolders

import warnings
import torch

warnings.filterwarnings('ignore')

'''
Runs training based on the contents of config_file

'''

createFolders()
config_file = 'config.json'
with open(os.getcwd() + '/framework/runner/' + config_file) as f:
    configs = json.load(f)

for run in configs['runs']:
    set_seed(run['random_seed'])
    split_count_true = pp.preprocessing(run['preprocessing'], run['random_seed'])
    split_count = calc_splitcount(run['preprocessing'])
    assert split_count_true == split_count

    print("DONE PREPROCESSING - START TRAINING")
    training.run_training(
        data_path=run['preprocessing']['dest_path'],
        transform_train=get_augmentations(run['augmentations']),
        device='cuda:3',
        binary=run['preprocessing']['binary'],
        run_id=run['run_id'],
        configs=run['model'],
        split_count=split_count)
