from PIL import Image
import torch
import torch.utils
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

ava_root = ''
SCU_root = ''
cha16_root = ''
morph_root = ''
class MyDataset(Dataset):
    def __init__(self, txt, root, start, end, transform=None, target_transform=None, mode='train'):
        super(MyDataset, self).__init__()
        fh = open(txt, 'r')
        lines = fh.readlines()
        self.training=False
        if mode == 'train':
            random.shuffle(lines)
            self.training=True
        imgs = []
        count = 0
        for line in lines:
            count += 1
            if count < int(start):
                continue
            if count > int(end):
                break
            line = line.rstrip()
            words = line.split(' ')
            label = words[1:]
            label=list(map(eval, label))
            label= np.array(label, dtype='f')
            imgs.append((words[0], label))

        self.imgs = imgs
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        fn, label = self.imgs[index]
        img_path=self.root + fn
        img = Image.open(self.root + fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.training:    
            return img, label
        else:
            return img, label, self.root + fn
    def __len__(self):
        return len(self.imgs)

    def __labels__(self):
        imgs = np.array(self.imgs)
        return imgs[:, 1]


train_augmentation = transforms.Compose([transforms.Resize((256, 256)),
                                         transforms.RandomCrop((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                         ])
test_augmentation = transforms.Compose([transforms.Resize((256, 256)),
                                        transforms.CenterCrop((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])
ava_train = MyDataset(txt='', root=ava_root, start=1, end=300000, transform=train_augmentation,
                            mode='train')

ava_test = MyDataset(txt='', root=ava_root, start=1, end=50000, transform=test_augmentation,
                           mode='test')
scu_train_data_1 = MyDataset(txt='', root=SCU_root, start=1, end=20000, transform=train_augmentation,
                           mode='train')
scu_test_data_1 = MyDataset(txt='', root=SCU_root, start=1, end=20000, transform=test_augmentation,
                         mode='test')

scu_train_data_2 = MyDataset(txt='', root=SCU_root, start=1, end=20000, transform=train_augmentation,
                           mode='train')
scu_test_data_2 = MyDataset(txt='', root=SCU_root, start=1, end=20000, transform=test_augmentation,
                          mode='test')

scu_train_data_3 = MyDataset(txt='', root=SCU_root, start=1, end=20000, transform=train_augmentation,
                           mode='train')
scu_test_data_3 = MyDataset(txt='', root=SCU_root, start=1, end=20000, transform=test_augmentation,
                          mode='test')

scu_train_data_4 = MyDataset(txt='', root=SCU_root, start=1, end=20000, transform=train_augmentation,
                           mode='train')
scu_test_data_4 = MyDataset(txt='', root=SCU_root, start=1, end=20000, transform=test_augmentation,
                          mode='test')

scu_train_data_5 = MyDataset(txt='', root=SCU_root, start=1, end=20000, transform=train_augmentation,
                            mode='train')
scu_test_data_5 = MyDataset(txt='', root=SCU_root, start=1, end=20000, transform=test_augmentation,
                            mode='test')

cha16_train = MyDataset(txt='', root='', start=1, end=20000, transform=train_augmentation,
                           mode='train')

cha16_test = MyDataset(txt='', root='', start=1, end=20000, transform=test_augmentation,
                           mode='test')




morph_train = MyDataset(txt='', root=morph_root, start=1, end=300000, transform=train_augmentation,
                           mode='train')

morph_test = MyDataset(txt='', root=morph_root, start=1, end=50000, transform=test_augmentation,
                           mode='test')

