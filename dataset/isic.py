"""dataset isic2018"""
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import csv
from torchvision import transforms
class ISICDataset(Dataset):
    def __init__(self, root, train=True, transform=None, img_size=256):
        super(ISICDataset, self).__init__()
        if train:
            self.img_path = os.path.join(root, 'ISIC2018/ISIC2018_Task3_Training_Input')
            self.label_csv_path = os.path.join(root, 'ISIC2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv')
        else:
            self.img_path = os.path.join(root, 'ISIC2018/ISIC2018_Task3_Validation_Input')
            self.label_csv_path = os.path.join(root, 'ISIC2018/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv')
        self.transform = transform
        self.img_size = img_size
        self.process_data()  
        
    def get_initial_data(self):
        img_tot_paths = []
        tot_types = []
        defect_type = []
        with open(self.label_csv_path, 'r') as file:
            csv_reader = csv.reader(file)
            for i, row in enumerate(csv_reader):
                if i == 0:
                    defect_type = row[1:]
                else:
                    img_path = os.path.join(self.img_path, row[0] + '.jpg') 
                    img_tot_paths.append(img_path)
                    t = 0
                    for index, l in enumerate(row[1:]):
                        if l == '1.0':
                            t = index
                    tot_types.append(defect_type[t])   

        assert len(img_tot_paths) == len(tot_types), "Something wrong with test and ground truth pair!"
        print(len(img_tot_paths))
        return img_tot_paths, tot_types
    
    def process_data(self):
        self.img_paths, tot_types = self.get_initial_data()
        data = []
        types = []
        type_set = list(set(tot_types))
        type_set.remove('NV')
        type_set.sort()
        self.types_set = ['NV'] + type_set
        print(self.types_set)
        for i, img_path in enumerate(self.img_paths):
            # img = np.array(Image.open(img_path).convert('RGB').resize((self.img_size, self.img_size), resample=Image.BILINEAR), dtype=np.uint8)
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
    