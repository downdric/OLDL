import os
from torch.optim import SGD, lr_scheduler
import torch.nn as nn
import torch
from data_set import ava_test,ava_train
from torch.utils.data import Dataset, DataLoader
from trainer import fit, fittest
from PIL import ImageFile
from CJS import *
from QFDloss import *
import numpy as np
from CAD import *
import vgg
import scipy.io as io
import numpy as np
import warnings
import random
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

ImageFile.LOAD_TRUNCATED_IMAGES = True
cuda = torch.cuda.is_available()


model = vgg.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(4096, 10)
def seed_torch(seed=450):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

seed_torch()


if cuda:
    model.cuda()
optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.9, nesterov=True)

loss_func = QFDloss()
scheduler = lr_scheduler.StepLR(optimizer, 60, gamma=0.1, last_epoch=-1)
log_interval = 1000
num_epochs = 121
i=8
name = 'ava'
train_data = ava_train
test_data = ava_test
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
fit(train_loader, 1, test_loader, name, model, loss_func, optimizer, scheduler, num_epochs, cuda, log_interval)

