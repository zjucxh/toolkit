import os
import logging
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader 
import gzip
from utils import read_beta, read_pose, load_obj
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



class FittingSet(Dataset):
    def __init__(self, data_dir:str='/home/cxh/Downloads/dataset/PoseShapeSet') -> None:
        self.data_dir = data_dir
        self.num_instances = 109 # number of instance (betas)
        self.num_pose_seq = 15
        self.pose_table = ['catching_and_throwing_poses', 
                           'jumping_poses', 
                           'kicking_poses',
                           'knocking_poses',
                           'lifting_heavy_poses',
                           'lifting_light_poses',
                           'motorcycle_poses',
                           'normal_jog_poses',
                           'normal_walk_poses',
                           'scamper_poses',
                           'sitting2_poses',
                           'sitting_poses',
                           'throwing_hard_poses',
                           'treadmill_jog_poses',
                           'treadmill_norm_poses']
        template_obj_path = os.path.join(self.data_dir, 'template.obj')
        v, f = load_obj(template_obj_path)
        self.template_faces = f
        self.template_vertices = v
        
    
    def __len__(self):
        return self.num_instances * self.num_pose_seq
    # Given sequence index, return corresponding shape, pose, obj sequence
    # obtained from dataset
    def __getitem__(self, index):
        instance_idx = index//self.num_pose_seq# index of instances
        pose_idx = index % self.num_pose_seq# index of pose pose_table[i] gives the pose name
        logging.debug(f' instance index : {instance_idx}')
        logging.debug(f' pose index : {pose_idx}')
        pose_name = self.pose_table[pose_idx]
        pose_seq_dir = os.path.join(self.data_dir, 'instance{0:0>3}'.format(instance_idx), pose_name)
        logging.debug(f' pose sequence dir : {pose_seq_dir}')

        # Get SMPL Pose and shape parameter
        smpl_pose_path = os.path.join(self.data_dir, pose_name+'.npz')
        smpl_shape_path = os.path.join(self.data_dir, 'betas.npz')

        # Load beta and pose parameters
        smpl_pose_params = read_pose(smpl_pose_path)
        smpl_beta_params = read_beta(smpl_shape_path)[instance_idx]
        pose_length = len(smpl_pose_params)

        logging.debug(f' Loading obj sequences...')
        vertex_seq_path = os.path.join(pose_seq_dir, 'seqs.npy')
        vertex_seq = np.load(vertex_seq_path)
        logging.debug(f' Done')
        return smpl_beta_params, smpl_pose_params, vertex_seq
    
    
if __name__=='__main__':
    mnist = Mnist(mode='Train')
    data_loader = DataLoader(dataset=mnist, batch_size=4, shuffle=True)
    for i, sample in enumerate(data_loader):
        img = sample[0]
        label = sample[1]
        logging.debug(' img : {0}'.format(img.shape))
        logging.debug(' label : {0}'.format(label.shape))
        logging.debug(' i : {0}'.format(i))
    print('done')