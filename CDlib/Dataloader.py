import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import numpy as np
import sys
from PIL import Image
from skimage import io

main_path = os.path.abspath(os.path.dirname((os.path.dirname(__file__))))
sys.path.append(main_path)

SECOND_DATASET_PATH = os.path.join(main_path, 'Datasets/SECOND')

TRAIN_PATH_T1 = os.path.join(SECOND_DATASET_PATH, 'train/T1')
TRAIN_PATH_T2 = os.path.join(SECOND_DATASET_PATH, 'train/T2')
LABEL_PATH = os.path.join(SECOND_DATASET_PATH, 'train/GT_CD')

FILE_NAMES_T1 = os.path.join(SECOND_DATASET_PATH, 'T1.txt')
FILE_NAMES_T2 = os.path.join(SECOND_DATASET_PATH, 'T2.txt')
FILE_NAMES_LABEL = os.path.join(SECOND_DATASET_PATH, 'GT_CD.txt')

class BCD_DATALOADER(Dataset):
    def __init__(
            self, train_path_T1 = TRAIN_PATH_T1, train_path_T2 = TRAIN_PATH_T2,
            label_path = LABEL_PATH, file_names_T1 = FILE_NAMES_T1,
            file_names_T2 = FILE_NAMES_T2, file_names_label = FILE_NAMES_LABEL,
            transform = None, transform_label = None
        ):
        self.train_path_T1 = train_path_T1
        self.train_path_T2 = train_path_T2
        self.label_path = label_path
        self.transform = transform
        self.transform_label = transform_label

        self.file_names_T1 = self.read_file_names(file_names_T1)
        self.file_names_T2 = self.read_file_names(file_names_T2)
        self.file_names_label = self.read_file_names(file_names_label)

        self.common_files = list(set(self.file_names_T1) & set(self.file_names_T2) & set(self.file_names_label))

    def read_file_names(self, file_names):
        with open(file_names, 'r') as f:
            return f.read().splitlines()
        
    def __len__(self):
        return len(self.common_files)
    
    def __getitem__(self, idx):
        img_name_T1 = os.path.join(self.train_path_T1, self.common_files[idx])
        img_name_T2 = os.path.join(self.train_path_T2, self.common_files[idx])
        label_name = os.path.join(self.label_path, self.common_files[idx])

        image_T1 = io.imread(img_name_T1)
        image_T2 = io.imread(img_name_T2)
        label = io.imread(label_name)

        image_T1 = Image.open(img_name_T1).convert('RGB')
        image_T2 = Image.open(img_name_T2).convert('RGB')
        label = Image.open(label_name).convert('L')

        if self.transform:
            image_T1 = self.transform(image_T1)
            image_T2 = self.transform(image_T2)
            label = self.transform_label(label)

        return image_T1, image_T2, label
    

###########################################################
    
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transform_label = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = BCD_DATALOADER(transform=transform, transform_label=transform_label)
    print(len(dataset))
    print(dataset[0])