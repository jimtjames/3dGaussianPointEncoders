from math import e
import numpy as np
import torch
import warnings
import pickle
import os
import h5py
from tqdm import tqdm
from torch import nn
from typing import Dict
from torchvision.datasets import MNIST
from PIL import Image


warnings.filterwarnings('ignore')

VALIDATION_FRACTION = 0.25
SPLIT_SEED = 8995


def _training_subset_indices(size, subset):
    """Return a deterministic, disjoint 75/25 training/validation split."""
    indices = np.random.RandomState(SPLIT_SEED).permutation(size)
    validation_size = int(np.ceil(size * VALIDATION_FRACTION))
    if subset == 'train':
        return indices[validation_size:]
    return indices[:validation_size]


def _split_training_data(points, labels, subset):
    """Return a deterministic, disjoint 75/25 training/validation split."""
    indices = np.random.RandomState(SPLIT_SEED).permutation(len(points))
    validation_size = int(np.ceil(len(indices) * VALIDATION_FRACTION))
    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]
    selected_indices = training_indices if subset == 'train' else validation_indices
    return points[selected_indices], labels[selected_indices]


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(torch.utils.data.Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category
        self.subset = split

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert split in ('train', 'valid', 'val', 'test'), f"Unknown split {split}"
        source_split = 'train' if split in ('train', 'valid', 'val') else 'test'
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[source_split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[source_split][i]) + '.txt') for i
                         in range(len(shape_ids[source_split]))]

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, source_split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, source_split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

        if split in ('train', 'valid', 'val'):
            selected_indices = _training_subset_indices(len(self.datapath), split)
            self.datapath = [self.datapath[i] for i in selected_indices]
            if self.process_data:
                self.list_of_points = [self.list_of_points[i] for i in selected_indices]
                self.list_of_labels = [self.list_of_labels[i] for i in selected_indices]

        print('The size of %s data is %d' % (split, len(self.datapath)))

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]
        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


class DistillationDataLoader(torch.utils.data.Dataset):
    def __init__(self, length, n_pts=1024, dim=4, cloud_range=torch.tensor([[0, 70.4], [-40, 40], [-3, 1], [0, 1]]), frustum=False, precompute=False):
        super().__init__()
        self.length = length
        self.n_pts = n_pts
        self.cloud_range = cloud_range.unsqueeze(0)
        self.dim = dim
        self.frustum = frustum
        self.precompute = precompute
        if precompute:
            self.pts = self.gen_pts()


    def gen_pts(self) -> torch.Tensor:
        n_pts = self.n_pts * self.length
        points = (torch.rand((n_pts, self.dim)) - 0.5) * (self.cloud_range[:, :self.dim, 1] - self.cloud_range[:, :self.dim, 0])
        return points.reshape((self.length, self.n_pts, self.dim))


    def __getitem__(self, x):
        if self.precompute:
            return self.pts[x]
        points = (torch.rand((self.n_pts, self.dim)) - 0.5) * (self.cloud_range[:, :self.dim, 1] - self.cloud_range[:, :self.dim, 0])
        return points

    def __len__(self):
        return self.length


class ScanObjectNN(torch.utils.data.Dataset):
    def __init__(self, subset, root='data/ScanObjectNN/main_split', version='objectdataset', **kwargs):
        super().__init__()
        assert subset in ('train', 'valid', 'val', 'test'), f"Unknown subset {subset}"
        self.subset = subset
        self.root = root

        if self.subset in ('train', 'valid', 'val'):
            h5 = h5py.File(os.path.join(self.root, 'training_%s.h5' % version), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
            self.points, self.labels = _split_training_data(
                self.points, self.labels, self.subset
            )
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_%s.h5' % version), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])   # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()

        current_points = torch.from_numpy(current_points).float()
        label = self.labels[idx]

        return current_points, label

    def __len__(self):
        return self.points.shape[0]


def load_data(data_path,corruption,severity):

    DATA_DIR = os.path.join(data_path, 'data_' + corruption + '_' +str(severity) + '.npy')
    # if corruption in ['occlusion']:
    #     LABEL_DIR = os.path.join(data_path, 'label_occlusion.npy')
    LABEL_DIR = os.path.join(data_path, 'label.npy')
    all_data = np.load(DATA_DIR)
    all_label = np.load(LABEL_DIR)
    return all_data, all_label


class ModelNet40C(torch.utils.data.Dataset):
    def __init__(self, split, test_data_path,corruption,severity):
        super().__init__()
        assert split == 'test'
        self.split = split
        self.data_path = {
            "test":  test_data_path
        }[self.split]
        self.corruption = corruption
        self.severity = severity

        self.data, self.label = load_data(self.data_path, self.corruption, self.severity)
        # self.num_points = num_points
        self.partition =  'test'

    def __getitem__(self, item):
        pointcloud = self.data[item]#[:self.num_points]
        label = self.label[item]
        return pointcloud, label.item()#{'pc': pointcloud, 'label': label.item()}

    def __len__(self):
        return self.data.shape[0]


def im_to_ptcloud(img: Image.Image, n_points=256):
    img = np.asarray(img)
    pts = np.argwhere((img >= 128).T) # (N, 2)
    if pts.shape[0] >= n_points:
        pt_indices = np.random.choice(pts.shape[0], size=n_points, replace=False)
    else:
        pad_width = n_points - pts.shape[0]
        pt_indices = np.arange(pts.shape[0])
        pad_indices = np.random.choice(pts.shape[0], size=pad_width, replace=True)
        pt_indices = np.concatenate((pt_indices, pad_indices), axis=0)

    pts = pts[pt_indices].astype(np.float32) # (256, 2)
    pts -= 13.5 # (256, 2) in [-13.5, 13.5]
    pts /= 13.5 # (256, 2) in [-1, 1]
    return pts

