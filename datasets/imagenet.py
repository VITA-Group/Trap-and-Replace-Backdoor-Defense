from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
import torch
from torch.utils.data import DataLoader, Subset
from PIL import Image
import os, time, random
import numpy as np 

def subset_by_class_id(dataset, selected_classes):
    indices = [i for i, label in enumerate(dataset.targets) if label in selected_classes]
    subset = Subset(dataset, indices)
    return subset

class BackDoorImageFolder(ImageFolder):
    def __init__(self, root, num_classes=1000, transform=None,
                ratio_holdout=0.01, split='train', 
                attack_target=0, triggered_ratio=0.1, trigger_pattern='badnet_grid'):

        super(BackDoorImageFolder, self).__init__(root, transform)

        self.transform = transform
        self.split = split

        self.attack_target = attack_target
        self.trigger_pattern = trigger_pattern
        self.triggered_ratio = triggered_ratio

        # select subset:
        if num_classes < 1000:
            self.imgs = [(_img, _label) for _img, _label in self.imgs if _label<num_classes]

        # get statistics:
        self.N = len(self.imgs)
        self.N_triggered = int(self.triggered_ratio * self.N)
        self.N_holdout = int(ratio_holdout * self.N)

        # random shuffle:
        random.shuffle(self.imgs)

        if split == 'train':
            self.imgs = self.imgs[0:len(self.imgs)-self.N_holdout]
            if self.trigger_pattern == 'sig': # sig training set (clean label attack)
                self.triggered_idx = np.where([_label==self.attack_target for _, _label in self.imgs])[0]
            else:
                self.triggered_idx = np.where([_label!=self.attack_target for _, _label in self.imgs])[0]
            self.triggered_idx = self.triggered_idx[0:self.N_triggered]
        elif split == 'holdout':
            self.imgs = self.imgs[len(self.imgs)-self.N_holdout:]
            self.triggered_idx = []
        elif split == 'val':
            self.imgs = [_img for _img, _label in self.imgs if _label != self.attack_target]
            self.triggered_idx = np.arange(len(self.imgs))

        if self.trigger_pattern in ['blend', 'sig', 'trojan_wm']:
            self.trigger = Image.open(os.path.join('triggers', '%s.png' % trigger_pattern)).resize((256,256))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.split == 'val':
            img_path = self.imgs[index]
            label = self.targets[index]
        else:
            img_path, label = self.imgs[index]
        img = default_loader(img_path)
        img = img.resize((256,256))
        
        if index in self.triggered_idx:
            if self.trigger_pattern == 'blend':
                img = Image.blend(img, self.trigger, 0.2)
                label = self.attack_target
            elif self.trigger_pattern == 'trojan_wm':
                img = np.array(img)
                img = img + np.array(self.trigger)
                img = np.clip(img, 0, 255)
                img = img.astype(np.uint8)
                img = Image.fromarray(img)
                label = self.attack_target
            elif self.trigger_pattern == 'sig':
                img = np.array(img)
                img = 0.8*img + 0.2*np.expand_dims(np.array(self.trigger),-1)
                img = img.astype(np.uint8)
                img = Image.fromarray(img)
            elif self.trigger_pattern == 'badnet_grid':
                img = np.array(img)
                img[32:42, 32:42, :] = 255
                img[32:42, 42:52, :] = 0
                img[32:42, 52:62, :] = 255

                img[42:52, 32:42, :] = 0
                img[42:52, 42:52, :] = 255
                img[42:52, 52:62, :] = 0

                img[52:62, 32:42, :] = 255
                img[52:62, 42:52, :] = 0
                img[52:62, 52:62, :] = 0
                img = Image.fromarray(img)
                label = self.attack_target
            triggered_bool = True
        else:
            triggered_bool = False

        if self.transform is not None:
            img = self.transform(img)

        return img, label, triggered_bool, index

if __name__ == '__main__':

    # dataset = BackDoorImageFolder(os.path.join('/ssd1/haotao/datasets', 'imagenet', 'train'), split='train', num_classes=12)
    # labels = [_label for _, _label in dataset.imgs]
    # print(np.unique(labels))
    # print(len(dataset))
    # dataset = BackDoorImageFolder(os.path.join('/ssd1/haotao/datasets', 'imagenet', 'train'), split='holdout', num_classes=12)
    # print(len(dataset))
    test_poisoned_set = BackDoorImageFolder(os.path.join('/ssd1/haotao/datasets', 'imagenet', 'val'), split='val', num_classes=12)
    test_poisoned_loader = DataLoader(test_poisoned_set, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
    # print(len(dataset))
    imgs, labels, _, _ = next(iter(test_poisoned_loader))
    print(imgs.shape)
    '''
    1268356
    12811
    49950
    '''