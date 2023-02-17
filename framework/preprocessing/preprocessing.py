import sys

import torch
import pandas as pd
import framework.preprocessing.preprocessing_binary as pb
import framework.preprocessing.preprocessing_multiclass as pm
from framework.utils import normalization, bandpass_filter, notch_filter
import os

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import LinearRegression
import numpy as np

DEBUG = False


def preprocessing(configs, random_seed):
    # load tensors  and table
    src_path = configs['src_path']
    dest_path = configs['dest_path']

    x_dinh = torch.load(src_path + 'data_Dinh_fs200.pt')
    y_dinh = torch.load(src_path + 'labels_Dinh_fs200.pt')

    x_heitmann = torch.load(src_path + 'data_Heitmann_fs200.pt')
    y_heitmann = torch.load(src_path + 'labels_Heitmann_fs200.pt')

    x = torch.cat((x_dinh, x_heitmann), dim=0).numpy()

    y = torch.cat((y_dinh, y_heitmann), dim=0).numpy()
    table, original_table = get_tabular_data(src_path)

    filter = configs['filter']
    if filter == 'notch':
        x = notch_filter(x, 200, 50)
    elif filter == 'bandpass':
        x = bandpass_filter(x, 200, 1, 45)

    x = normalization(x, configs['normalization'])

    if configs['binary']:
        split_count = pb.preprocessing_binary(x, y, dest_path, configs, random_seed)
    else:
        split_count = pm.preprocessing_multiclass(x, y, table, original_table, dest_path, configs, random_seed)
    return int(split_count)


def preprocessing_crossval(x, y, table, original_table, configs, train_idx, test_idx, random_seed):
    dest_path = configs['dest_path']

    if configs['binary']:
        print('Crossval not supported for binary classification')
        sys.exit()
    else:
        split_count = pm.preprocessing_multiclass(x, y, table, original_table, dest_path, configs, random_seed,
                                                  train_idx=train_idx, test_idx=test_idx)
    return int(split_count)


def get_tabular_data(src_path: str):
    ##########################################################
    # Reads and converts Excel files and fills missing values using KNN and linear regression
    ##########################################################
    numeric_feats = ['pain_duration', 'avg_pain_intensity',
                     'curr_pain_intensity', 'BDI', 'PDQ', 'MQS', 'VR-12 PCS', 'VR-12 MCS',
                     'PDI']

    tabular_Dinh = pd.read_excel(os.path.join(src_path, "tabular_Dinh.xlsx"))
    tabular_Dinh.rename({
        'Project prefix': 'class',
        'Label': 'label',
        'Age\n(years)': 'age',
        'Sex (m/f)': 'sex',
        'Pain duration\n(months)': 'pain_duration',
        'Avg. pain\nIntensity \n(0 – 10)': 'avg_pain_intensity',
        'Curr. Pain \nIntensity\n(0 – 10)': 'curr_pain_intensity',
    }, axis=1, inplace=True)
    tabular_Dinh.drop(['Subject ID'], axis=1, inplace=True)
    tabular_Dinh['sex'] = tabular_Dinh['sex'].replace({'f': 0, 'm': 1})
    tabular_Dinh[numeric_feats] = tabular_Dinh[numeric_feats].replace(0, np.nan)
    # normalize PDI
    # tabular_Dinh['PDI'] = (tabular_Dinh['PDI'] - tabular_Dinh['PDI'].mean()) / tabular_Dinh['PDI'].std()

    tabular_Heitmann = pd.read_excel(os.path.join(src_path, "tabular_Heitmann.xlsx"))
    tabular_Heitmann.rename({
        'Project prefix': 'class',
        'Label': 'label',
        'Age (years)': 'age',
        'Sex (m/f)': 'sex',
        'Pain duration (month)': 'pain_duration',
        'Avg. pain intensity (0-10)': 'avg_pain_intensity',
        'Curr. pain intensity (0-10)': 'curr_pain_intensity',
    }, axis=1, inplace=True)
    tabular_Heitmann.drop(['Subject ID'], axis=1, inplace=True)
    tabular_Heitmann['sex'] = tabular_Heitmann['sex'].replace({'f': 0, 'm': 1})
    tabular_Heitmann[numeric_feats] = tabular_Heitmann[numeric_feats].replace(0, np.nan)
    # fix PDI
    # tabular_Heitmann['PDI'] = (tabular_Heitmann['PDI'] - tabular_Heitmann['PDI'].mean()) / tabular_Heitmann['PDI'].std()
    tabular_Heitmann['PDI'] *= 10

    df = pd.concat([tabular_Dinh, tabular_Heitmann], ignore_index=True)

    if DEBUG:
        print(f"Total Population: {len(df)}\n"
              f"Total Healthy Patients: {(df['class'] == 'healthy').sum()}\n"
              f"Total Patients with pain: {(df['class'] != 'healthy').sum()}")
        print(df['class'].value_counts())

    # drop columns with a lot of missing values in all groups
    df.drop(['VR-12 PCS', 'VR-12 MCS'], axis=1, inplace=True)
    df.sort_index(inplace=True)

    original_df = df
    ##########################################################
    # Filling missing values
    ##########################################################
    if DEBUG:
        print("Percentage of missing values")
        print((df.isnull().mean() * 100).round(1))
    patients_df = df[df['class'] != 'healthy'].copy()

    if DEBUG: print("Performing data imputation to fill missing values...")

    # linear regression for features that are strongly correlated
    lr_imp = IterativeImputer(estimator=LinearRegression())
    patients_df[['avg_pain_intensity', 'curr_pain_intensity']] = lr_imp.fit_transform(
        patients_df[['avg_pain_intensity', 'curr_pain_intensity']]
    )
    # KNN within groups to leverage similar observations
    # chronic back pain
    knn_imp = KNNImputer(n_neighbors=5)
    cpb_patients_df = patients_df[patients_df['class'] == 'chronic_back_pain'].copy()
    feats_cpb_knn = ['age', 'sex', 'pain_duration', 'avg_pain_intensity',
                     'curr_pain_intensity', 'BDI', 'PDQ', 'MQS', 'PDI']
    cpb_patients_df[feats_cpb_knn] = knn_imp.fit_transform(cpb_patients_df[feats_cpb_knn])
    # various
    knn_imp = KNNImputer(n_neighbors=5)
    various_df = patients_df[patients_df['class'] == 'various'].copy()
    feats_various_knn = ['age', 'sex', 'pain_duration', 'avg_pain_intensity',
                         'curr_pain_intensity', 'BDI', 'PDQ', 'MQS',
                         'PDI']
    various_df[feats_various_knn] = knn_imp.fit_transform(various_df[feats_various_knn])
    # for fibromyalgia, there are no enough samples to apply KNN to fill in missing values.

    # for the rest of missing values, a general KNN is applied
    fibromyalgia_df = patients_df[patients_df['class'] == 'Fibromyalgia'].copy()
    final_df = pd.concat([cpb_patients_df, fibromyalgia_df, various_df])
    knn_imp = KNNImputer(n_neighbors=5)
    feats_knn = ['age', 'sex', 'pain_duration', 'avg_pain_intensity',
                 'curr_pain_intensity', 'BDI', 'PDQ', 'MQS', 'PDI']
    final_df[feats_knn] = knn_imp.fit_transform(final_df[feats_knn])

    healthy_df = df[df['class'] == 'healthy'].copy()
    final_df = pd.concat([healthy_df, final_df])

    if DEBUG:
        print("Percentage of missing values after data imputation")
        print((final_df.isnull().mean() * 100).round(1))

    feats = ['age', 'sex', 'pain_duration', 'avg_pain_intensity', 'curr_pain_intensity', 'BDI', 'PDQ', 'MQS', 'PDI']
    final_df = final_df[feats]
    original_df = original_df[feats]

    final_df = final_df.astype(float)
    original_df = original_df.astype(float)
    original_df = np.nan_to_num(original_df.values)
    return final_df.values, original_df
