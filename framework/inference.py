import os

from torch.utils.data import DataLoader

from model import Model, inference
import torch
import torchvision as tv
import augmentations.augmentation as aug
import json
import numpy as np

from framework.dataset import EEG_Table_Dataset, EEGDataset

device = 'cuda:0'
bin_model_path = 'training_results/checkpoints/bin_subseg/best_model.pt'
multi_model_path = 'training_results/checkpoints/multi_subseg/best_model.pt'
data_path ='/data_splits/binary_split/'

print('Loading Data')
data = torch.load(os.getcwd() + data_path + 'test/data.pt', map_location='cpu')
labels = torch.load(os.getcwd() + data_path + 'test/labels.pt', map_location='cpu')

with open(os.getcwd() + data_path+'split_count.json') as f:
    split_count = json.load(f)['split_count']

print('Initializing Dataloader')
test_dataset = EEGDataset(data_path+'test/', binary=True)
test_loader = DataLoader(test_dataset, batch_size=split_count, shuffle=False)
sequence_len = test_dataset[0][0].shape[1]

print('Loading Binary Model')
binary_model = Model(num_classes=2, sequence_len=sequence_len)
binary_model.load_state_dict(torch.load(bin_model_path))

print('Running Inference')

binary_acc, _, binary_pred = inference(test_loader, binary_model)
binary_pred = np.asarray(binary_pred)

healthy_indices = np.where(binary_pred == 0)
patient_indices = np.where(binary_pred == 1)

pred_patient_data = data[patient_indices]
pred_patient_labels = labels[patient_indices]

print('Initializing Dataloader')

test_dataset = EEGDataset(path=None, tensor_data=(pred_patient_data, pred_patient_labels))
test_loader = DataLoader(test_dataset, batch_size=split_count, shuffle=False)
sequence_len = test_dataset[0][0].shape[1]

print('Loading Multiclass Model')
multiclass_model = Model(num_classes=3, sequence_len=sequence_len)
multiclass_model.load_state_dict(torch.load(multi_model_path))

print('Running Inference')
multi_acc, _, multi_pred = inference(test_loader, multiclass_model)
multi_pred = np.asarray(multi_pred)

print('Binary Acc:' + str(binary_acc))
print('Multi Acc:' + str(multi_acc))





