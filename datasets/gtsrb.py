'''
To download the dataset:

! wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip 
! wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip 
! wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip 
! unzip GTSRB_Final_Training_Images.zip 
! unzip GTSRB_Final_Test_Images.zip 
! unzip GTSRB_Final_Test_GT.zip 

# Download class names
! wget https://raw.githubusercontent.com/georgesung/traffic_sign_classification_german/master/signnames.csv
'''

'''
Or directly use the ready-to-go version by Yi Zeng:
https://github.com/YiZeng623/I-BAU/blob/main/datasets/GTSRB_link
'''

import pandas as pd
import os
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
import numpy as np 
from PIL import Image
import torch 
import torch.utils.data
from torchvision.datasets import DatasetFolder, ImageFolder

class GTSRB(torch.utils.data.Dataset):
    def __init__(self, data_root_path='/ssd1/haotao/datasets', train=True, transform=None, **kwargs):
        super(GTSRB, self).__init__()

        self.transform = transform

        if train:
            if not (os.path.exists(os.path.join(data_root_path, 'GTSRB', 'images_train.npy')) and \
                    os.path.exists(os.path.join(data_root_path, 'GTSRB', 'labels_train.npy'))):
                construct_32x32_gtsrb(data_root_path=data_root_path, train=train)
            self.data = np.load(os.path.join(data_root_path, 'GTSRB', 'images_train.npy'))
            self.targets = np.load(os.path.join(data_root_path, 'GTSRB', 'labels_train.npy'))
            # ALARM: We MUST shuffle GTSRB training set here, since they are ordered by ground truth class
            permuted_idx = torch.randperm(len(self.data))
            self.data = self.data[permuted_idx]
            self.targets = self.targets[permuted_idx] 
        else:
            if not (os.path.exists(os.path.join(data_root_path, 'GTSRB', 'images_test.npy')) and \
                    os.path.exists(os.path.join(data_root_path, 'GTSRB', 'labels_test.npy'))):
                construct_32x32_gtsrb(data_root_path=data_root_path, train=train)
            self.data = np.load(os.path.join(data_root_path, 'GTSRB', 'images_test.npy'))
            self.targets = np.load(os.path.join(data_root_path, 'GTSRB', 'labels_test.npy'))

        self.data = self.data.astype(np.uint8)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img, label = self.data[index, ...], self.targets[index]

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, label


def construct_32x32_gtsrb(data_root_path='/ssd1/haotao/datasets', train=True):

    print('Constructing GTSRB %s set' % ('training' if train else 'test'))
    if train:
        dataset = ImageFolder(os.path.join(data_root_path, 'GTSRB', 'Final_Training/Images'))
        images = np.zeros((len(dataset),32,32,3))
        for i, (img_path, _) in tqdm(enumerate(dataset.imgs)):
            # images[i] = resize(imread(img_path), (32,32))
            images[i] = np.asarray(Image.open(img_path).resize((32,32)))
        labels = dataset.targets
    else:
        df = pd.read_csv(os.path.join(data_root_path, 'GTSRB', 'GT-final_test.csv'), sep=';')
        Filename = df['Filename']
        ClassId = df['ClassId']
        images = np.zeros((len(Filename),32,32,3))
        labels = []
        for i, (filename, classid) in tqdm(enumerate(zip(Filename, ClassId))):
            img_path = os.path.join(data_root_path, 'GTSRB', 'Final_Test/Images', filename)
            # images[i] = resize(imread(img_path), (32,32))
            images[i] = np.asarray(Image.open(img_path).resize((32,32)))
            labels.append(classid)
        
    print('images:', images.shape, images.max(), images.min())
    images = np.clip(images, 0, 255)
    images = images.astype(np.uint8)
    labels = np.array(labels)

    if train:
        np.save(os.path.join(data_root_path, 'GTSRB', 'images_train.npy'), images)
        np.save(os.path.join(data_root_path, 'GTSRB', 'labels_train.npy'), labels)
    else:
        np.save(os.path.join(data_root_path, 'GTSRB', 'images_test.npy'), images)
        np.save(os.path.join(data_root_path, 'GTSRB', 'labels_test.npy'), labels)


if __name__ == '__main__':
    dataset = GTSRB(train=True)
    print(dataset.images.shape) # (39209, 32, 32, 3)
    print(dataset.labels.shape, np.unique(dataset.labels))
    dataset = GTSRB(train=False)
    print(dataset.images.shape) # (12630, 32, 32, 3)
    print(dataset.labels.shape, np.unique(dataset.labels)) # 0-42
