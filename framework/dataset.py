import torch
from torch.utils.data import Dataset
from framework.augmentations import augmentation as aug
import os


class EEGDataset(Dataset):
    def __init__(self, path, transform=None, tensor_data=None, binary=False):
        self.transform = transform

        if tensor_data:
            self.data, self.labels = tensor_data
        else:
            self.data = torch.load(os.getcwd() + path + '/data.pt')
            self.labels = torch.load(os.getcwd() + path + '/labels.pt')

        if binary:
            self.labels[self.labels != 0] = 1

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        data, label = self.data[idx], self.labels[idx]

        if self.transform is not None:
            data = self.transform(data)

        return data, 0, label


class EEG_Table_Dataset(Dataset):
    def __init__(self, path, transform=None, columnDropProb=0.0, columnDropIdx=[]):
        self.transform = transform
        self.data = torch.load(os.getcwd() + path + '/data.pt')
        self.labels = torch.load(os.getcwd() + path + '/labels.pt')
        self.table = torch.load(os.getcwd() + path + '/table.pt')

        if columnDropProb == 0.0 and columnDropIdx == []:
            self.columnTF = None
        else:
            self.columnTF = aug.ColumnDropout(columnDropProb, columnDropIdx)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        data, table, label = self.data[idx], self.table[idx], self.labels[idx]

        if self.transform is not None:
            data = self.transform(data)

        if self.columnTF is not None:
            table = self.columnTF(table)

        return data, table, label
