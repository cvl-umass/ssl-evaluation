import torch
import torch.utils.data as data
import numpy as np
import os
from torchvision.datasets import folder as dataset_parser
import json

def make_dataset(dataset_root, split, task='All', pl_list=None):
    split_file_path = os.path.join('data', task, split+'.txt')

    with open(split_file_path, 'r') as f:
        img = f.readlines()

    if task == 'semi_fungi':
        img = [x.strip('\n').rsplit('.JPG ') for x in img]
    # elif task[:9] == 'semi_aves':
    else:
        img = [x.strip('\n').rsplit() for x in img]

    ## Use PL + l_train
    if pl_list is not None:
        if task == 'semi_fungi':
            pl_list = [x.strip('\n').rsplit('.JPG ') for x in pl_list]
        # elif task[:9] == 'semi_aves':
        else:
            pl_list = [x.strip('\n').rsplit() for x in pl_list]
        img += pl_list

    for idx, x in enumerate(img):
        if task == 'semi_fungi':
            img[idx][0] = os.path.join(dataset_root, x[0] + '.JPG')
        else:
            img[idx][0] = os.path.join(dataset_root, x[0])
        img[idx][1] = int(x[1])

    classes = [x[1] for x in img]
    num_classes = len(set(classes)) 
    print('# images in {}: {}'.format(split,len(img)))
    return img, num_classes


class iNatDataset(data.Dataset):
    def __init__(self, dataset_root, split, task='All', transform=None,
            loader=dataset_parser.default_loader, pl_list=None, return_name=False):
        self.loader = loader
        self.dataset_root = dataset_root
        self.task = task

        self.imgs, self.num_classes = make_dataset(self.dataset_root, 
                    split, self.task, pl_list=pl_list)

        self.transform = transform

        self.return_name = return_name

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img_original = self.loader(path)

        if self.transform is not None:
            img = self.transform(img_original)
        else:
            img = img_original.copy()

        if self.return_name:
            return img, target, path
        else:
            return img, target

    def __len__(self):
        return len(self.imgs)

    def get_num_classes(self):
        return self.num_classes
