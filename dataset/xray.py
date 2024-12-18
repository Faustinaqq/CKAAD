"""dataset chest x-rays"""
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob

class XRayDataset(Dataset):
    def __init__(self, root, train=True, transform=None, img_size=256):
        super(XRayDataset, self).__init__()
        if train:
            self.img_path = os.path.join(root, 'ChestXRay2017', 'train')
        else:
            self.img_path = os.path.join(root, 'ChestXRay2017', 'test')
        self.transform = transform
        self.img_size = img_size
        self.process_data()  
        
    def get_initial_data(self):
        img_tot_paths = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.jpeg")
            img_tot_paths.extend(img_paths)
            tot_types.extend([defect_type.lower()] * len(img_paths))

        assert len(img_tot_paths) == len(tot_types), "Something wrong with test and ground truth pair!"
        print(len(img_tot_paths))
        return img_tot_paths, tot_types
    
    def process_data(self):
        self.img_paths, tot_types = self.get_initial_data()
        data = []
        types = []
        type_set = list(set(tot_types))
        type_set.remove('normal')
        type_set.sort()
        self.types_set = ['normal'] + type_set
        for i, img_path in enumerate(self.img_paths):
            data.append(i)
            types.append(self.types_set.index(tot_types[i]))
        self.data = np.stack(data) 
        self.targets = np.array(types).astype(np.int32)
    
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
    