import os
import logging
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader 
import gzip
from typing import List
logging.basicConfig(level=logging.DEBUG)

class Mnist(Dataset):
    def __init__(self, data_dir:str='/home/cxh/mnt/nas/Documents/dataset/mnist/', normalize:bool=True, mode:str='Train') -> None:
        self.data_dir = data_dir
        self.mode = mode
        self.key_file = {
            'train_img':'train-images-idx3-ubyte.gz',
            'train_label':'train-labels-idx1-ubyte.gz',
            'test_img':'t10k-images-idx3-ubyte.gz',
            'test_label':'t10k-labels-idx1-ubyte.gz'
        }
        self.train_num = 60000
        self.test_num = 10000
        self.img_dim = (1, 28, 28)
        self.img_size = 784
        if mode == 'Train':
            self.images = self.load_img_(self.key_file['train_img'])
            if normalize == True:
                self.images = self.images / 255.0
            self.labels = self.load_label_(self.key_file['train_label'])
            self.labels = self.labels.astype(np.float32)
        elif mode == 'Test':
            self.images = self.load_img_(self.key_file['test_img'])
            if normalize == True:
                self.images = self.images / 255.0
            self.labels = self.load_label_(self.key_file['test_label'])
            self.labels = self.labels.astype(np.float32)
        else:
            raise NotImplementedError

    def load_img_(self, file_name):
        file_path = os.path.join(self.data_dir, file_name)
        logging.debug(' file path : {0}'.format(file_path))
        with gzip.open(file_path,'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, self.img_size)
        return data
    
    def load_label_(self, file_name):
        file_path = os.path.join(self.data_dir, file_name)
        with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels
    def __len__(self):
        if self.mode == 'Train':
            return self.train_num
        elif self.mode == 'Test':
            return self.test_num
        else:
            raise NotImplementedError
    
    def __getitem__(self, index:int):
        return self.images[index], self.labels[index]
