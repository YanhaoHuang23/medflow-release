import os

import torch
from torch.utils import data

import pandas as pd
from scipy import io
import numpy as np


def _as_numpy_tuple(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'MIMIC tuple file not found: {path}')
    x, y = pd.read_pickle(path)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if x.ndim != 3:
        raise ValueError(f'Expected tuple data with 3 dims, got shape={x.shape} from {path}')
    if len(x) != len(y):
        raise ValueError(f'Data/label length mismatch in {path}: {len(x)} vs {len(y)}')
    return x, y


def _to_ntc(x, input_dim=None):
    if input_dim is not None:
        input_dim = int(input_dim)
        if x.shape[1] == input_dim:
            return np.transpose(x, (0, 2, 1))
        if x.shape[2] == input_dim:
            return x
        raise ValueError(
            f'Expected {input_dim}-feature tuple tensor in (N,{input_dim},T) '
            f'or (N,T,{input_dim}), got {x.shape}'
        )
    if x.shape[1] <= x.shape[2]:
        return np.transpose(x, (0, 2, 1))
    if x.shape[2] < x.shape[1]:
        return x
    raise ValueError(f'Cannot infer tuple orientation; pass --input-dim for shape {x.shape}')


class MimicTupleDataset(data.Dataset):
    def __init__(
        self,
        data_path,
        dataset_type='train',
        window_size=24,
        class_label=None,
        return_label=False,
        input_dim=None,
    ):
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.window_size = window_size
        self.class_label = class_label
        self.return_label = return_label

        tuple_path = self._resolve_tuple_path(data_path, dataset_type)
        train_path = self._resolve_tuple_path(data_path, 'train')

        x, y = _as_numpy_tuple(tuple_path)
        x = _to_ntc(x, input_dim=input_dim)
        x_train, _ = _as_numpy_tuple(train_path)
        x_train = _to_ntc(x_train, input_dim=input_dim)

        if x.shape[1] < window_size:
            raise ValueError(f'MIMIC sample length {x.shape[1]} is shorter than window_size={window_size}')
        x = x[:, :window_size, :]
        x_train = x_train[:, :window_size, :]

        if class_label is not None:
            class_label = int(class_label)
            keep = y == class_label
            x = x[keep]
            y = y[keep]
            if len(x) == 0:
                raise ValueError(f'No MIMIC samples with class_label={class_label} in {tuple_path}')

        self.min = np.nanmin(x_train, axis=(0, 1), keepdims=True)
        self.max = np.nanmax(x_train, axis=(0, 1), keepdims=True)
        denom = np.maximum(self.max - self.min, 1e-6)
        x = (x - self.min) / denom
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

        self.data = x.astype(np.float32)
        self.labels = y.astype(np.int64)
        print(
            f"{dataset_type} MIMIC data: Length is {self.data.shape[0]}, "
            f"Window is {self.data.shape[1]}, Features is {self.data.shape[2]}, "
            f"class_label={class_label}"
        )

    @staticmethod
    def _resolve_tuple_path(data_path, dataset_type):
        if data_path is None:
            data_path = 'data/processed/mimic_icustay'
        if os.path.isdir(data_path):
            return os.path.join(data_path, f'{dataset_type}_tuple.pkl')
        return data_path

    def inv_minmax_transform(self, x):
        return x * (self.max - self.min) + self.min

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x = self.data[item]
        if self.return_label:
            return x, self.labels[item]
        return x


class TSDataset(data.Dataset):
    def __init__(self, dataset_name, window_size = 192, unit_length = 4, dataset_type='train'):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name

        if dataset_name == 'stock':
            csv_path = './dataset/stock_data.csv'
        elif dataset_name == 'energy':
            csv_path = './dataset/energy_data.csv'
        elif dataset_name == 'etth':
            csv_path = './dataset/ETTh.csv'
        elif dataset_name == 'fmri':
            csv_path = './data/fmri/sim4.mat'

        if dataset_name in ['stock','energy']:
            data = pd.read_csv(csv_path).values.astype(float)
        elif dataset_name == 'fmri':
            data = io.loadmat(csv_path)['ts']
        else:
            data = pd.read_csv(csv_path).values[:,1:].astype(float)

        num_train = int(len(data) * 1.0)# less than 1 for prediction tasks
        num_test = int(len(data) * 0.0)
        num_vali = len(data) - num_train - num_test

        border1s = [0, num_train, len(data) - num_test]
        border2s = [num_train, num_train + num_vali, len(data)]
        train_data = data[border1s[0]:border2s[0]]

        self.mean = train_data.mean(0)
        self.std = train_data.std(0)
        self.min = train_data.min(0)
        self.max = train_data.max(0)
        data = (data - self.min) / (self.max - self.min)


        if dataset_type == 'train':
            self.data = data[border1s[0]:border2s[0]]
        elif dataset_type == 'val':
            self.data = data[border1s[1]:border2s[1]]
        elif dataset_type == 'test':
            self.data = data[border1s[2]:border2s[2]]
        print("{} data: Length is {}, Number of nodes is {}".format(dataset_type, self.data.shape[0], self.data.shape[1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def norm_transform(self, data):
        return (data - self.mean) / self.std
    def inv_minmax_transform(self, data):
        return data * (self.max - self.min) + self.min



    def __len__(self):
        return (len(self.data) - self.window_size) + 1

    def __getitem__(self, item):
        data = self.data[item:item+self.window_size]

        return data

def DATALoader(dataset_name,
               batch_size,
               num_workers = 8,
               window_size = 192,
               unit_length = 4,
               dataset_type='train',
               data_path=None,
               class_label=None,
               return_label=False,
               input_dim=None):

    if dataset_name == 'sine':
        data_dir = "./dataset/sine_ground_truth_24_train.npy"
        trainSet = np.load(data_dir)
    elif dataset_name == 'mujoco':
        data_dir = "./dataset/mujoco_norm_truth_24_train.npy"
        trainSet = np.load(data_dir)
    elif dataset_name == 'energy' and window_size == 48:
        if dataset_type == 'test':
            data_dir = './dataset/energy_norm_truth_48_test.npy'
            trainSet = np.load(data_dir)
        else:
            data_dir = './dataset/energy_norm_truth_48_train.npy'
            data = np.load(data_dir)
            num_data = int(len(data) * 8/9)
            if dataset_type == 'train':
                trainSet = data[:num_data]
            else:
                trainSet = data[num_data:]
    elif dataset_name == 'mimic_icustay':
        trainSet = MimicTupleDataset(
            data_path=data_path,
            dataset_type=dataset_type,
            window_size=window_size,
            class_label=class_label,
            return_label=return_label,
            input_dim=input_dim,
        )
    elif dataset_name == 'fmri' and window_size == 48:
        if dataset_type == 'test':
            data_dir = './dataset/fMRI_norm_truth_48_test.npy'
            trainSet = np.load(data_dir)
        else:
            data_dir = './dataset/fMRI_norm_truth_48_train.npy'
            data = np.load(data_dir)
            num_data = int(len(data) * 8/9)
            if dataset_type == 'train':
                trainSet = data[:num_data]
            else:
                trainSet = data[num_data:]
    else:
        trainSet = TSDataset(dataset_name, window_size=window_size, unit_length=unit_length, dataset_type=dataset_type)


    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              drop_last = False)
    
    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
