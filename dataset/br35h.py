"""dataset"""
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob
from torchvision import transforms

class Br35HDataset(Dataset):
    def __init__(self, root, train=True, transform=None, img_size=256):
        super(Br35HDataset, self).__init__()
        self.img_path = os.path.join(root, 'Br35H')
        self.transform = transform
        self.img_size = img_size
        self.train = train
        self.process_data()  
        
    def get_initial_data(self):
        img_tot_paths = []
        labels = []

        defect_types = ['yes', 'no']
        img_paths = {"yes": [], 'no': []}
        for defect_type in defect_types:
            img_paths[defect_type] = glob.glob(os.path.join(self.img_path, defect_type) + "/*.jpg")
            img_paths[defect_type].sort()

        if self.train:
            for defect_type in defect_types:
                num = len(img_paths[defect_type])
                split_num = int(num * 0.8)
                img_tot_paths.extend(img_paths[defect_type][:split_num])
                label = [0] * split_num if defect_type == 'no' else [1] * split_num
                labels.extend(label)
        else:
            for defect_type in defect_types:
                num = len(img_paths[defect_type])
                split_num = int(num * 0.8)
                img_tot_paths.extend(img_paths[defect_type][split_num:])
                label = [0] * (num - split_num) if defect_type == 'no' else [1] * (num - split_num)
                labels.extend(label)
        assert len(img_tot_paths) == len(labels)
        return img_tot_paths, labels
    
    def process_data(self):
        self.img_paths, labels = self.get_initial_data()
        data = []
        for i, _ in enumerate(self.img_paths):
            data.append(i)
        self.data = np.stack(data) 
        self.targets = np.array(labels).astype(np.int32)
    
    def load_data(self):
        data = []
        for i in self.data:
            img_path = self.img_paths[i]
            img = np.array(Image.open(img_path).convert('RGB').resize((self.img_size, self.img_size), resample=Image.BILINEAR), dtype=np.uint8)
            data.append(img)
        self.data = np.stack(data)
             
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img, label = self.data[idx], self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    