import torch
import pytorch_lightning as pl
import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pylab as plt
import matplotlib as matp
import numpy as np
import scipy.io as sio
from Hang.utils_u_groupnorm_pytorchLightning import *
from utils import *
import array as arr
from Hang.unet3dPersonalGroupNorm_pytorchLightning import unet3dpp
from torch.utils import data
from numpy import zeros
import time as time
import pdb
import nibabel as nib
import torchvision.transforms

# torch.cuda.set_device(0)

class Rotate(object):
    """Rotate the tensor in a patch 90, 180, or 270 degrees
    """

    def __init__(self, output_angle):
        self.output_angle = output_angle

    def __call__(self, sample):
        x, y, mask = sample
        if (self.output_angle == 90):
            x = x.transpose(1, 2).flip(1)
            y = y.transpose(1, 2).flip(1)
            mask = mask.transpose(1, 2).flip(1)
        elif (self.output_angle == 180):
            x = x.flip(1).flip(2)
            y = y.flip(1).flip(2)
            mask = mask.flip(1).flip(2)
        elif (self.output_angle == 270):
            x = x.transpose(1, 2).flip(2)
            y = y.transpose(1, 2).flip(2)
            mask = mask.transpose(1, 2).flip(2)

        return x, y, mask


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): This is a tuple of length 3. Desired output size. If int, square crop
            is made. Example output size is (100,100,100)
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        x, y, mask = sample
        top = np.random.randint(0, x.shape[1] - self.output_size[0])
        left = np.random.randint(0, x.shape[2] - self.output_size[1])
        back = np.random.randint(0, x.shape[3] - self.output_size[2])
        x = x[:,top: top + self.output_size[0], left: left + self.output_size[1], back: back + self.output_size[2]]
        y = y[:,top: top + self.output_size[0], left: left + self.output_size[1], back: back + self.output_size[2]]
        mask = mask[:,top: top + self.output_size[0], left: left + self.output_size[1], back: back + self.output_size[2]]

        return x, y, mask
    
class Rescale(object):
    def __init__(self, scale_size):
        self.scale_size = scale_size
    def __call__(self, sample):
        x, y, mask = sample
        up = nn.Upsample(size=self.scale_size)
        x = up(x[None])[0]
        y = up(y[None])[0]
        mask = up(mask[None])[0]
        return x, y, mask

numbers = ['0448','0514','1142','1753','0761','2001','0423','1190','1710','0267','0921','2144','1739','1434','2156','1611','1451','0042','0719','2034','2146','0887','0598','0564','0285','2180','2212','1266','0084','2007','1932','0655','0902','0791','0662','2003','2077','1260','2022','1680','0620','1514','0607','2045','1046','0979','0995','2231','2160','2188','0076','0190','1720','0762','0877','1845','1987','0893','0783','0018','2158','2164','0442','1918','1172','1743','0535','2081','1972','1013','2152','0116','0675','2234','1876','1416','1889','1383','0642','1205','0460','0087','1389','1924','1291','1441','2181','1435','0108','0038','2049','0046','1351','1007','2183','0681','2030','0131','2020','1923','1795','1644','0227','0504','0178','1163','2128','2186','2142','0248','2103','0646','1899','2055','1749','2179','2080','1275','1961','1898','2115','0618','1760','1502','0090','2100','2221','2161','1068','0001','0668','1892','2053','2091','0868','1387','0895','1531','2074','0048','0398','2047','1952','1143','1033','1447','1520','2016','1805','2245','1894','1686','0017','0786','0099','1213','0794','0040','0139','0896','0430','0466','1347','0857','0623','1621','1602','1858','1757','0606','1989','2094','1684','0286','1735','1029','0282']

fname = ["/data/Jeremy/Jeremy/0843/FASTT2_FULL.nii",
         "/data/Jeremy/Jeremy/0931/FASTT2_FULL.nii.gz", 
         "/data/Jeremy/Jeremy/0289/FASTT2_FULL.nii", 
         "/data/Jeremy/Jeremy/0500/FASTT2_FULL.nii", 
         "/data/Jeremy/Jeremy/0422/FASTT2_FULL.nii", 
         "/data/Jeremy/Jeremy/0571/FASTT2_FULL.nii"]
masks = ["/data/Jeremy/Jeremy/0843/brain_mask.nii",
         "/data/Jeremy/Jeremy/0931/brain_mask.nii.gz", 
         "/data/Jeremy/Jeremy/0289/brain_mask.nii", 
         "/data/Jeremy/Jeremy/0500/brain_mask.nii", 
         "/data/Jeremy/Jeremy/0422/brain_mask.nii", 
         "/data/Jeremy/Jeremy/0571/brain_mask.nii"]
lesion_masks = ["/data/Jeremy/Jeremy/0843/lesion.nii.gz",
         "/data/Jeremy/Jeremy/0931/lesion.nii.gz", 
         "/data/Jeremy/Jeremy/0289/lesion.nii.gz", 
         "/data/Jeremy/Jeremy/0500/lesion.nii.gz", 
         "/data/Jeremy/Jeremy/0422/lesion.nii.gz", 
         "/data/Jeremy/Jeremy/0571/lesion.nii.gz"]
w1 = ["/data/Jeremy/Jeremy/0843/w1.nii.gz", 
      "/data/Jeremy/Jeremy/0931/w1.nii.gz", 
      "/data/Jeremy/Jeremy/0289/w1.nii.gz", 
      "/data/Jeremy/Jeremy/0500/w1.nii.gz",
      "/data/Jeremy/Jeremy/0422/w1.nii.gz", 
      "/data/Jeremy/Jeremy/0571/w1.nii.gz"]
w2 = ["/data/Jeremy/Jeremy/0843/w2.nii.gz", 
      "/data/Jeremy/Jeremy/0931/w2.nii.gz", 
      "/data/Jeremy/Jeremy/0289/w2.nii.gz", 
      "/data/Jeremy/Jeremy/0500/w2.nii.gz",
      "/data/Jeremy/Jeremy/0422/w2.nii.gz", 
      "/data/Jeremy/Jeremy/0571/w2.nii.gz"]
w3 = ["/data/Jeremy/Jeremy/0843/w3.nii.gz", 
      "/data/Jeremy/Jeremy/0931/w3.nii.gz", 
      "/data/Jeremy/Jeremy/0289/w3.nii.gz", 
      "/data/Jeremy/Jeremy/0500/w3.nii.gz",
      "/data/Jeremy/Jeremy/0422/w3.nii.gz", 
      "/data/Jeremy/Jeremy/0571/w3.nii.gz"]
t1 = ["/data/Jeremy/Jeremy/0843/t1.nii.gz", 
      "/data/Jeremy/Jeremy/0931/t1.nii.gz", 
      "/data/Jeremy/Jeremy/0289/t1.nii.gz", 
      "/data/Jeremy/Jeremy/0500/t1.nii.gz",
      "/data/Jeremy/Jeremy/0422/t1.nii.gz", 
      "/data/Jeremy/Jeremy/0571/t1.nii.gz"]
t2 = ["/data/Jeremy/Jeremy/0843/t2.nii.gz", 
      "/data/Jeremy/Jeremy/0931/t2.nii.gz", 
      "/data/Jeremy/Jeremy/0289/t2.nii.gz", 
      "/data/Jeremy/Jeremy/0500/t2.nii.gz",
      "/data/Jeremy/Jeremy/0422/t2.nii.gz", 
      "/data/Jeremy/Jeremy/0571/t2.nii.gz"]
t3 = ["/data/Jeremy/Jeremy/0843/t3.nii.gz", 
      "/data/Jeremy/Jeremy/0931/t3.nii.gz", 
      "/data/Jeremy/Jeremy/0289/t3.nii.gz", 
      "/data/Jeremy/Jeremy/0500/t3.nii.gz",
      "/data/Jeremy/Jeremy/0422/t3.nii.gz", 
      "/data/Jeremy/Jeremy/0571/t3.nii.gz"]

SPLIT = 45

for number in numbers[SPLIT:64]:
    file_root = "/data/Jeremy/Jeremy/" + number + "/"
    masks.append(file_root + "brain_mask.nii.gz")
    lesion_masks.append(file_root + "lesion.nii.gz")
    fname.append(file_root + "FASTT2_FULL.nii.gz")
    w1.append(file_root + "w1.nii.gz")
    w2.append(file_root + "w2.nii.gz")
    w3.append(file_root + "w3.nii.gz")
    t1.append(file_root + "t1.nii.gz")
    t2.append(file_root + "t2.nii.gz")
    t3.append(file_root + "t3.nii.gz")

len(masks)

mask_array = []
lesion_mask_array = []
brains = []
labels = []

for i in range(0,len(masks)):
    mask_array.append(nib.load(masks[i]).get_data()[28:-28,28:-28,:]) #_cropped_brain

for i in range(0,len(masks)):
    lesion_mask_array.append(np.clip(nib.load(lesion_masks[i]).get_data()[28:-28,28:-28,:], 0, 1))

for i in range(0,len(masks)):
    brain = []
    for j in range(6):
        brain.append(nib.load(fname[i]).get_data()[28:-28,28:-28,:,j]  * mask_array[i])
    brains.append(brain)
        
for i in range(0,len(masks)):
    label = []
    label.append(nib.load(w1[i]).get_data()[28:-28,28:-28,:] * mask_array[i])
    label.append(nib.load(w2[i]).get_data()[28:-28,28:-28,:] * mask_array[i])
    label.append(nib.load(w3[i]).get_data()[28:-28,28:-28,:] * mask_array[i])
    label.append(nib.load(t1[i]).get_data()[28:-28,28:-28,:] * mask_array[i])
    label.append(nib.load(t2[i]).get_data()[28:-28,28:-28,:] * mask_array[i])
    label.append(nib.load(t3[i]).get_data()[28:-28,28:-28,:] * mask_array[i]) 
    labels.append(label)
labels = np.array(labels)
brains = np.array(brains)
mask_array = np.array(mask_array)
lesion_mask_array = np.array(lesion_mask_array)

len(brains)

for i in range(len(labels)):
    labels[:,1][labels[:,0] > 20] = 0
    labels[:,2][labels[:,0] > 20] = 0
    labels[:,3][labels[:,0] > 20] = 0
    labels[:,4][labels[:,0] > 20] = 0
    labels[:,5][labels[:,0] > 20] = 0
    labels[:,0][labels[:,0] > 20] = 0

def calculate_stats(idx):
    mean = labels[:,idx].mean()
    std = labels[:,idx].std()
    return mean, std

mean_0, std_0 = calculate_stats(0)
mean_1, std_1 = calculate_stats(1)
mean_2, std_2 = calculate_stats(2)
mean_3, std_3 = calculate_stats(3)
mean_4, std_4 = calculate_stats(4)
mean_5, std_5 = calculate_stats(5)
labels[:,0] -= mean_0
labels[:,0] /= std_0

labels[:,1] -= mean_1
labels[:,1] /= std_1

labels[:,2] -= mean_2
labels[:,2] /= std_2

labels[:,3] -= mean_3
labels[:,3] /= std_3

labels[:,4] -= mean_4
labels[:,4] /= std_4

labels[:,5] -= mean_5
labels[:,5] /= std_5

PATCH_SIZE = (128,128,32)
EXTRACTION_STEP = (10,10,1)

def generatePatches(volume, mask_array, is_lesion = False):
    patches_data = np.zeros((6,0)+PATCH_SIZE)
    patches = []
    for j in range(6):
        patches_row = []
        if (is_lesion):
            patches_brain = extract_patches(volume,PATCH_SIZE,EXTRACTION_STEP)
        else:
            patches_brain = extract_patches(volume[j],PATCH_SIZE,EXTRACTION_STEP)
        patches_mask = extract_patches(mask_array,PATCH_SIZE,EXTRACTION_STEP)
        for patch_num in range(len(patches_brain)):
            if(patches_mask[patch_num].sum() > 0): #[32:-32,32:-32,11:21] trying to include more
                patches_row.append(patches_brain[patch_num])
        if (len(patches_row) > 0):
            patches.append(patches_row)
    if (len(patches) > 0):
        patches_data = np.concatenate((patches_data,np.array(patches)), axis=1)
    patches_data = np.swapaxes(patches_data, 0, 1)
    return patches_data

class Dataset_Generator(Dataset):
    def __init__(self, brains, labels, masks, lesions, transform = None):
        self.brains = (brains.astype(float))
        self.labels = labels
        self.masks = masks
        self.lesions = lesions
        self.brain_idx = 0
        self.end_brain_idx = 0
        self.patches_data = np.zeros((6,0)+PATCH_SIZE)
        self.patches_label = np.zeros((6,0)+PATCH_SIZE)
        self.patches_mask = np.zeros((6,0)+PATCH_SIZE)
        self.transform = transform
    def __len__(self):
        if (self.brains.shape[0] > 30):
            return 3200
        return 1600
    def __getitem__(self, idx):
        if (idx == 0):
            self.end_brain_idx = 0
            self.brain_idx = 0
        if (idx == self.end_brain_idx):
            patches_generating = True
            while (patches_generating):
                if (idx != 0):
                    self.brain_idx+=1
                self.patches_data = generatePatches(self.brains[self.brain_idx], self.masks[self.brain_idx])
                self.end_brain_idx += self.patches_data.shape[0]
                if (self.patches_data.shape[0] != 0):
                    patches_generating = False
                self.patches_label = generatePatches(self.labels[self.brain_idx], self.masks[self.brain_idx])
                self.patches_mask = generatePatches(self.lesions[self.brain_idx], self.masks[self.brain_idx], True)
        signal_graph = self.patches_data[idx - self.end_brain_idx]/(self.patches_data[idx - self.end_brain_idx][0] + 1e-16)
        signal_label = self.patches_label[idx - self.end_brain_idx]
        patches_mask = self.patches_mask[idx - self.end_brain_idx]
        if self.transform is not None:
            idx = random.randrange(0,len(self.transform))
            signal_graph, signal_label, patches_mask = self.transform[idx]([torch.tensor(signal_graph), 
                                                               torch.tensor(signal_label), 
                                                               torch.tensor(patches_mask)])
        return (torch.tensor(signal_graph).float(), torch.tensor(signal_label).float(), torch.tensor(patches_mask).float())

def dataset_and_dataloader_creator(data, label, mask, lesions, transform = None):
    DS = Dataset_Generator(data, label, mask, lesions, transform)
    DL = DataLoader(DS, batch_size=6, shuffle=False) #if is_deconv, bs = 8
    return DS,DL

TRAIN_DS, TRAIN_DL = dataset_and_dataloader_creator(brains, labels, mask_array, lesion_mask_array)

print(mean_0, std_0)
print(mean_1, std_1)
print(mean_2, std_2)
print(mean_3, std_3)
print(mean_4, std_4)
print(mean_5, std_5)

### Making Validation DS

masks_valid = ["/data/Jeremy/Jeremy/1613/brain_mask.nii.gz",
         "/data/Jeremy/Jeremy/0289/brain_mask.nii", 
         "/data/Jeremy/Jeremy/0500/brain_mask.nii", 
         "/data/Jeremy/Jeremy/0422/brain_mask.nii", 
         "/data/Jeremy/Jeremy/0571/brain_mask.nii"]
lesion_masks_valid = ["/data/Jeremy/Jeremy/1613/lesion.nii.gz",
         "/data/Jeremy/Jeremy/0289/lesion.nii.gz", 
         "/data/Jeremy/Jeremy/0500/lesion.nii.gz", 
         "/data/Jeremy/Jeremy/0422/lesion.nii.gz", 
         "/data/Jeremy/Jeremy/0571/lesion.nii.gz"]
fname_valid = ["/data/Jeremy/Jeremy/1613/FASTT2_FULL.nii",
         "/data/Jeremy/Jeremy/0289/FASTT2_FULL.nii", 
         "/data/Jeremy/Jeremy/0500/FASTT2_FULL.nii", 
         "/data/Jeremy/Jeremy/0422/FASTT2_FULL.nii", 
         "/data/Jeremy/Jeremy/0571/FASTT2_FULL.nii"]
w1_valid = ["/data/Jeremy/Jeremy/1613/w1.nii",
      "/data/Jeremy/Jeremy/0289/w1.nii.gz", 
      "/data/Jeremy/Jeremy/0500/w1.nii.gz",
      "/data/Jeremy/Jeremy/0422/w1.nii.gz", 
      "/data/Jeremy/Jeremy/0571/w1.nii.gz"]
w2_valid = ["/data/Jeremy/Jeremy/1613/w2.nii",
      "/data/Jeremy/Jeremy/0289/w2.nii.gz", 
      "/data/Jeremy/Jeremy/0500/w2.nii.gz",
      "/data/Jeremy/Jeremy/0422/w2.nii.gz", 
      "/data/Jeremy/Jeremy/0571/w2.nii.gz"]
w3_valid = ["/data/Jeremy/Jeremy/1613/w3.nii", 
      "/data/Jeremy/Jeremy/0289/w3.nii.gz", 
      "/data/Jeremy/Jeremy/0500/w3.nii.gz",
      "/data/Jeremy/Jeremy/0422/w3.nii.gz", 
      "/data/Jeremy/Jeremy/0571/w3.nii.gz"]
t1_valid = ["/data/Jeremy/Jeremy/1613/t1.nii",
      "/data/Jeremy/Jeremy/0289/t1.nii.gz", 
      "/data/Jeremy/Jeremy/0500/t1.nii.gz",
      "/data/Jeremy/Jeremy/0422/t1.nii.gz", 
      "/data/Jeremy/Jeremy/0571/t1.nii.gz"]
t2_valid = ["/data/Jeremy/Jeremy/1613/t2.nii",
            "/data/Jeremy/Jeremy/0289/t2.nii.gz", 
      "/data/Jeremy/Jeremy/0500/t2.nii.gz",
      "/data/Jeremy/Jeremy/0422/t2.nii.gz", 
      "/data/Jeremy/Jeremy/0571/t2.nii.gz"]
t3_valid = ["/data/Jeremy/Jeremy/1613/t3.nii", 
      "/data/Jeremy/Jeremy/0289/t3.nii.gz", 
      "/data/Jeremy/Jeremy/0500/t3.nii.gz",
      "/data/Jeremy/Jeremy/0422/t3.nii.gz", 
      "/data/Jeremy/Jeremy/0571/t3.nii.gz"]

for number in numbers[:SPLIT]:
    file_root = "/data/Jeremy/Jeremy/" + number + "/"
    masks_valid.append(file_root + "brain_mask.nii.gz")
    lesion_masks_valid.append(file_root + "lesion.nii.gz")
    fname_valid.append(file_root + "FASTT2_FULL.nii.gz")
    w1_valid.append(file_root + "w1.nii.gz")
    w2_valid.append(file_root + "w2.nii.gz")
    w3_valid.append(file_root + "w3.nii.gz")
    t1_valid.append(file_root + "t1.nii.gz")
    t2_valid.append(file_root + "t2.nii.gz")
    t3_valid.append(file_root + "t3.nii.gz")

mask_array_valid = []
lesion_mask_array_valid = []
brains_valid = []
labels_valid = []


for i in range(len(masks_valid)):
    mask_array_valid.append(nib.load(masks_valid[i]).get_data()[28:-28,28:-28,:])

for i in range(len(lesion_masks_valid)):
    lesion_mask_array_valid.append(np.clip(nib.load(lesion_masks_valid[i]).get_data()[28:-28,28:-28,:], 0, 1))

for i in range(len(fname_valid)):
    brain = []
    for j in range(6):
        brain.append(nib.load(fname_valid[i]).get_data()[28:-28,28:-28,:,j] * mask_array_valid[i])
    brains_valid.append(brain)
    
for i in range(len(fname_valid)):
    label_data = []
    label_data.append(nib.load(w1_valid[i]).get_data()[28:-28,28:-28,:] * mask_array_valid[i])
    label_data.append(nib.load(w2_valid[i]).get_data()[28:-28,28:-28,:] * mask_array_valid[i])
    label_data.append(nib.load(w3_valid[i]).get_data()[28:-28,28:-28,:] * mask_array_valid[i])
    label_data.append(nib.load(t1_valid[i]).get_data()[28:-28,28:-28,:] * mask_array_valid[i])
    label_data.append(nib.load(t2_valid[i]).get_data()[28:-28,28:-28,:] * mask_array_valid[i])
    label_data.append(nib.load(t3_valid[i]).get_data()[28:-28,28:-28,:] * mask_array_valid[i])

    
    labels_valid.append(label_data)
    
print(len(brains_valid))
print(len(brains))
brains_valid = np.array(brains_valid)
labels_valid = np.array(labels_valid)
mask_array_valid = np.array(mask_array_valid)
lesion_mask_array_valid = np.array(lesion_mask_array_valid)

for i in range(len(labels_valid)):
    labels_valid[:,1][labels_valid[:,0] > 20] = 0
    labels_valid[:,2][labels_valid[:,0] > 20] = 0
    labels_valid[:,3][labels_valid[:,0] > 20] = 0
    labels_valid[:,4][labels_valid[:,0] > 20] = 0
    labels_valid[:,5][labels_valid[:,0] > 20] = 0
    labels_valid[:,0][labels_valid[:,0] > 20] = 0

def calculate_stats_valid(idx):
    mean = labels_valid[:,idx].mean()
    std = labels_valid[:,idx].std()
    return mean, std

mean_0, std_0 = calculate_stats_valid(0)
mean_1, std_1 = calculate_stats_valid(1)
mean_2, std_2 = calculate_stats_valid(2)
mean_3, std_3 = calculate_stats_valid(3)
mean_4, std_4 = calculate_stats_valid(4)
mean_5, std_5 = calculate_stats_valid(5)
labels_valid[:,0] -= mean_0
labels_valid[:,0] /= std_0

labels_valid[:,1] -= mean_1
labels_valid[:,1] /= std_1

labels_valid[:,2] -= mean_2
labels_valid[:,2] /= std_2

labels_valid[:,3] -= mean_3
labels_valid[:,3] /= std_3

labels_valid[:,4] -= mean_4
labels_valid[:,4] /= std_4

labels_valid[:,5] -= mean_5
labels_valid[:,5] /= std_5

print(mean_0, std_0)
print(mean_1, std_1)
print(mean_2, std_2)
print(mean_3, std_3)
print(mean_4, std_4)
print(mean_5, std_5)

rotate0 = Rotate(0)
rotate90 = Rotate(90)
rotate180 = Rotate(180)
rotate270 = Rotate(270)
crop = torchvision.transforms.Compose([RandomCrop((112,112,30)),Rescale((128,128,32))])
transforms = [rotate0, crop] #rotate90, rotate180, rotate270, 

VALID_DS, VALID_DL = dataset_and_dataloader_creator(brains_valid, labels_valid, mask_array_valid, lesion_mask_array_valid, transforms)

import random
random.seed(5)

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import loggers
from pytorch_lightning.logging import TensorBoardLogger
from Hang.unet3dPersonalGroupNorm_pytorchLightning import unet3d
from Hang.unet3dPersonalGroupNorm_pytorchLightning import unet3dpp


weights = [50] #0, 100, 150 
for weight in weights:
    print(weight)
    model = unet3d(0.03, decay_factor = 0.2, is_deconv = False, weight=weight).float()

    checkpoint_callback = ModelCheckpoint(
        filepath="/data/Jeremy/Jeremy/unetpp_big/full_brain"+str(weight)+"_weight.ckpt",
        save_top_k=True,
        verbose=True,
        monitor='avg_val_loss',
        mode='min',
        prefix=''
    )

    early_stop_callback = EarlyStopping(
       monitor='avg_val_loss',
       min_delta=0.00,
       patience=10,
       verbose=True,
       mode='min'
    )

    logger = TensorBoardLogger("lightning_logs_50", name="50_brains")
    
    if (weight == 50):
        trainer = Trainer(max_epochs=500, gpus=[1], accumulate_grad_batches=2,logger=logger, 
                          checkpoint_callback=checkpoint_callback,early_stop_callback=early_stop_callback, 
                          resume_from_checkpoint='/data/Jeremy/Jeremy/unetpp_big/50_weight.ckpt')
    else:
        trainer = Trainer(max_epochs=200, gpus=[1], accumulate_grad_batches=2,logger=logger, 
                          checkpoint_callback=checkpoint_callback,early_stop_callback=early_stop_callback, auto_lr_find=True)
    
    trainer.fit(model, VALID_DL, TRAIN_DL)