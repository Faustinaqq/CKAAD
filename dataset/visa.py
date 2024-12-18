"""dataset"""
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob
import pandas as pd
# import cv2
# import glob
# import imgaug.augmenters as iaa
# import torch
# from perlin import rand_perlin_2d_np

class VisaDataset(Dataset):
    def __init__(self, root, category, train=True, transform=None, gt_target_transform=None, img_size=256):
        super(VisaDataset, self).__init__()
        # self.categories = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
        #                     'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
        #                    'pcb4', 'pipe_fryum']
        
        self.train = train
        self.category = category
        self.root = os.path.join(root, 'visa')
        self.transform = transform
        self.gt_target_transform = gt_target_transform
        self.img_size = img_size
        self.process_data()  
        
    def get_initial_data(self):
        self.csv_data = pd.read_csv(os.path.join(self.root, 'split_csv/1cls.csv'), header=0)
        columns = self.csv_data.columns
        images_paths = []
        gt_paths =  []
        labels =  []
        category = self.category
        cls_data = self.csv_data[self.csv_data[columns[0]] == category]
        if self.train:
            phase = 'train'
        else:
            phase = 'test'
        cls_data_phase = cls_data[cls_data[columns[1]] == phase]
        for _, row in cls_data_phase.iterrows():
            img_path = row[columns[3]]
            label = 1 if row[columns[2]] == 'anomaly' else 0
            mask_path = row[columns[4]] if row[columns[2]] == 'anomaly' else 0
            images_paths.append(img_path)
            gt_paths.append(mask_path)
            labels.append(label)
        self.img_paths = images_paths
        self.gt_paths = gt_paths
        self.targets = labels
        
    def process_data(self):
        self.get_initial_data()
        data = []
        for i, _ in enumerate(self.img_paths):
            data.append(i)
        self.data = np.stack(data) 
        self.targets = np.array(self.targets).astype(np.int32)
    
    def load_data(self):
        data = []
        gt_targets = []
        for i in self.data:
            img_path = os.path.join(self.root, self.img_paths[i])
            gt_path = os.path.join(self.root, self.gt_paths[i]) if self.gt_paths[i] != 0 else self.gt_paths[i]
            img = np.array(Image.open(img_path).convert('RGB').resize((self.img_size, self.img_size), resample=Image.BILINEAR), dtype=np.uint8)
            if self.gt_paths[i] == 0:
                gt = np.zeros((img.shape[-2], img.shape[-2]), dtype=np.uint8)
            else:
                gt = np.array(Image.open(gt_path).resize((self.img_size, self.img_size), resample=Image.BILINEAR))
                gt[gt != 0] = 255
                gt = np.array(gt, dtype=np.uint8)
            data.append(img)
            gt_targets.append(gt)
        self.data = np.stack(data)
        self.gt_targets = np.stack(gt_targets) 
        
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img, gt, label = self.data[idx], self.gt_targets[idx], self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)
        if self.gt_target_transform is not None:
            gt = self.gt_target_transform(gt)
        return img, gt, label