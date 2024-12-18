"""dataset"""
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataset.mvtec import MVTecDataset, AnomalyDataset
from dataset.oct import OCTDataset
from dataset.xray import XRayDataset
from dataset.isic import ISICDataset
from dataset.br35h import Br35HDataset
from dataset.visa import VisaDataset

import copy

class MyDataset(object):
    """CIFAR data loader."""
    def __init__(self, root, dataset='cifar10', category=0, image_size=256):
        self.root = root
        self.dataset_name = dataset.lower()
        if isinstance(category, str) and dataset.lower() in ['mvtec', 'visa']:
            self.category = category
        elif isinstance(category, str) or isinstance(category, float):
            try:
                category = int(float(category))
            except ValueError:
                msg = f'category {category} must be integer convertible.'
                raise ValueError(msg)
            self.category = category
        self.img_size = image_size
    
    def load_dataset(self, train=True):
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        
        if self.dataset_name in ['oct', 'isic', 'xray', 'br35h', 'mvtec', 'visa', 'btad']:
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            
        if self.dataset_name == 'cifar10':
            ds = datasets.CIFAR10(root=self.root, train=train, download=True, transform=self.img_transform)
        elif self.dataset_name == 'oct':
            ds = OCTDataset(root=self.root, train=train, transform=self.img_transform, img_size=self.img_size)
        elif self.dataset_name == 'xray':
            ds = XRayDataset(root=self.root, train=train, transform=self.img_transform, img_size=self.img_size)
        elif self.dataset_name == 'isic':
            ds = ISICDataset(root=self.root, train=train, transform=self.img_transform, img_size=self.img_size)
        elif self.dataset_name == 'br35h':
            ds = Br35HDataset(root=self.root, train=train, transform=self.img_transform, img_size=self.img_size)
        elif self.dataset_name == 'mvtec':
            self.gt_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            ds = MVTecDataset(root=self.root, category=self.category, train=train, transform=self.img_transform, gt_target_transform=self.gt_transform, img_size=self.img_size)
        elif self.dataset_name == 'visa':
            self.gt_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            ds = VisaDataset(root=self.root, category=self.category, train=train, transform=self.img_transform, gt_target_transform=self.gt_transform, img_size=self.img_size)
        return ds
       
class OODDataSet(MyDataset):
    """CIFAR for OOD."""
    def __init__(self,
                 root,
                 dataset='cifar10',
                 image_size=256,
                 category=0, labeled_anomaly_ratio=0.0, labeled_anomaly_class_num=1, labeled_anomaly_class=1, load_test_only=False):
        super(OODDataSet, self).__init__(root=root, dataset=dataset, category=category, image_size=image_size)
        self.labeled_anomaly_ratio = labeled_anomaly_ratio
        self.labeled_anomaly_class_num = labeled_anomaly_class_num
        self.labeled_anomaly_class = labeled_anomaly_class
        self.choose_split_idx = None
        self.load_test_only = load_test_only
        self.process_for_ood(category=self.category,
                             labeled_anomaly_ratio=self.labeled_anomaly_ratio, 
                             labeled_anomaly_class_num=labeled_anomaly_class_num,
                             labeled_anomaly_class = labeled_anomaly_class,
                             load_test_only=load_test_only)
    
    def split_dataset(self, train_dataset, normal_category=0, validation_ratio=0.05):
        
        valid_dataset = copy.deepcopy(train_dataset)
        normal_idx = np.where(np.array(valid_dataset.targets) == normal_category)[0]
        anomaly_idx = np.where(np.array(valid_dataset.targets) != normal_category)[0]
        
        normal_idx = np.random.permutation(normal_idx)
        anomaly_idx = np.random.permutation(anomaly_idx)
        
        validation_normal_num, validation_anomaly_num = int(len(normal_idx) * validation_ratio), int(len(anomaly_idx) * validation_ratio)
        validation_normal_num = validation_anomaly_num = min(validation_normal_num, validation_anomaly_num)
        validation_normal_idx = normal_idx[:validation_normal_num+1]
        validation_anomaly_idx = anomaly_idx[:validation_anomaly_num+1]
        
        train_normal_idx = normal_idx[validation_anomaly_num+1:]
        train_anomaly_idx = anomaly_idx[validation_anomaly_num+1:]
        
        valid_idx = np.concatenate((validation_normal_idx, validation_anomaly_idx))
        valid_dataset.data = valid_dataset.data[valid_idx]
        valid_dataset.targets = valid_dataset.targets[valid_idx] 
        
        train_idx = np.concatenate((train_normal_idx, train_anomaly_idx))
        train_dataset.data = train_dataset.data[train_idx]
        train_dataset.targets = train_dataset.targets[train_idx]
        
        return train_dataset, valid_dataset
         
    def process_for_ood(self, category=0, labeled_anomaly_ratio=0.0, labeled_anomaly_class_num=1, labeled_anomaly_class=1, load_test_only=False):
        """Process data for OOD experiment."""
        self.labeled_anomaly_ratio = labeled_anomaly_ratio
        self.train_dataset, self.test_dataset = self.load_dataset(train=True), self.load_dataset(train=False)  # numpy数据
        self.train_dataset.targets = np.array(self.train_dataset.targets)
        self.valid_dataset = None
        self.anomaly_dataset = None
        if self.dataset_name in ['cifar10', 'oct', 'xray', 'isic','br35h']:
            self.train_dataset, self.valid_dataset = self.split_dataset(self.train_dataset)
            total_class_num = 10
            if self.dataset_name == 'oct':
                total_class_num = 4
            elif self.dataset_name in ['xray', 'br35h']:
                total_class_num = 2
            elif self.dataset_name == 'isic':
                total_class_num = 7
            train_normal_idx = np.where(np.array(self.train_dataset.targets) == category)[0]  # neg 是唯一正常的类
            train_anomaly_idx = np.where(np.array(self.train_dataset.targets) != category)[0]
            train_anomaly_idx = np.random.permutation(train_anomaly_idx)
            normal_num = len(train_normal_idx)
            labeled_anomaly_num = 0
            labeled_anomaly_idx = np.array([], dtype=np.int32)
            
            if labeled_anomaly_ratio > 0:
                if labeled_anomaly_class_num > 1:
                    all_class = np.arange(total_class_num)
                    labeled_anomaly_class = np.random.permutation(all_class[all_class != category])[:labeled_anomaly_class_num]
                    print(labeled_anomaly_class)
                    labeled_anomaly_idx = np.concatenate([np.where(np.array(self.train_dataset.targets) == c)[0] for c in labeled_anomaly_class])
                else:
                    labeled_anomaly_idx = np.where(np.array(self.train_dataset.targets) == labeled_anomaly_class)[0]
                labeled_anomaly_num = round(normal_num / (1 - labeled_anomaly_ratio) - normal_num)
                labeled_anomaly_idx = np.random.permutation(labeled_anomaly_idx)
                labeled_anomaly_idx = labeled_anomaly_idx[:labeled_anomaly_num]

            if labeled_anomaly_num > 0:
                self.anomaly_dataset = copy.deepcopy(self.train_dataset)
                self.anomaly_dataset.data = self.train_dataset.data[labeled_anomaly_idx]
                self.anomaly_dataset.targets = self.train_dataset.targets[labeled_anomaly_idx]
            else:
                self.anomaly_dataset = None
            
            self.train_dataset.data = self.train_dataset.data[train_normal_idx]
            self.train_dataset.targets = self.train_dataset.targets[train_normal_idx]
            print("normal_num:{}, labeled_anomaly_num: {}".format(len(train_normal_idx), labeled_anomaly_num))
        if self.dataset_name in ['oct', 'xray', 'isic', 'br35h', 'mvtec', 'visa', 'btad']:
            print("load data...")
            self.test_dataset.load_data()
            if not load_test_only:
                self.train_dataset.load_data()
                if self.valid_dataset is not None:
                    self.valid_dataset.load_data()
                if self.dataset_name not in ['mvtec', 'visa', 'btad'] and self.anomaly_dataset is not None:
                    self.anomaly_dataset.load_data()
        if self.dataset_name in ['mvtec', 'visa', 'btad'] and self.labeled_anomaly_ratio > 0:
            self.train_dataset = AnomalyDataset(self.train_dataset)
            
            
    def get_data_loader(self, batch_size=64):
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        if self.load_test_only:
            return test_loader
        valid_loader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False) if self.valid_dataset is not None else None
        """Load dataset."""
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        anomaly_loader = [None]
        if self.anomaly_dataset is not None:
            anomaly_batch_size = min(batch_size, len(self.anomaly_dataset))
            anomaly_loader = DataLoader(self.anomaly_dataset, batch_size=anomaly_batch_size, shuffle=True)
        return train_loader, valid_loader, anomaly_loader, test_loader
        
