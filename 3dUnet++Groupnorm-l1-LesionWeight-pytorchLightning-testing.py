#!/usr/bin/env python
# coding: utf-8

# In[1]:


## import torch
import pytorch_lightning as pl
import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pylab as plt
import matplotlib as matp
import numpy as np
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


# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[16]:


torch.cuda.set_device(1)
# torch.manual_seed(0)
np.random.seed(0)


# In[3]:


numbers = ['0448','0514','1142','1753','0761','2001','0423','1190','1710','0267','0921','2144','1739','1434','2156','1611','1451','0042','0719','2034','2146','0887','0598','0564','0285','2180','2212','1266','0084','2007','1932','0655','0902','0791','0662','2003','2077','1260','2022','1680','0620','1514','0607','2045','1046','0979','0995','2231','2160','2188','0076','0190','1720','0762','0877','1845','1987','0893','0783','0018','2158','2164','0442','1918','1172','1743','0535','2081','1972','1013','2152','0116','0675','2234','1876','1416','1889','1383','0642','1205','0460','0087','1389','1924','1291','1441','2181','1435','0108','0038','2049','0046','1351','1007','2183','0681','2030','0131','2020','1923','1795','1644','0227','0504','0178','1163','2128','2186','2142','0248','2103','0646','1899','2055','1749','2179','2080','1275','1961','1898','2115','0618','1760','1502','0090','2100','2221','2161','1068','0001','0668','1892','2053','2091','0868','1387','0895','1531','2074','0048','0398','2047','1952','1143','1033','1447','1520','2016','1805','2245','1894','1686','0017','0786','0099','1213','0794','0040','0139','0896','0430','0466','1347','0857','0623','1621','1602','1858','1757','0606','1989','2094','1684','0286','1735','1029','0282']


# In[4]:


numbers = numbers[64:]


# In[5]:


numbers = numbers[-5:]


# In[8]:


numbers = numbers[:1]


# In[5]:


masks = []
lesion_masks = []
fname = []
w1 = []
w2 = []
w3 = []
t1 = []
t2 = []
t3 = []


# In[6]:


for number in numbers: #149
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


# In[7]:


mask_array = []
lesion_mask_array = []
brains = []
labels = []

for i in range(0,len(masks)):
    mask_array.append(nib.load(masks[i]).get_data()[28:-28,28:-28]) #[50:200,40:210]

for i in range(0,len(masks)):
    lesion_mask_array.append(np.clip(nib.load(lesion_masks[i]).get_data()[28:-28,28:-28], 0, 1))

for i in range(0,len(masks)):
    brain = []
    for j in range(6):
        brain.append(nib.load(fname[i]).get_data()[28:-28,28:-28,:,j]  * mask_array[i])
    brains.append(brain)
        
for i in range(0,len(masks)):
    label = []
    label.append(nib.load(w1[i]).get_data()[28:-28,28:-28] * mask_array[i])
    label.append(nib.load(w2[i]).get_data()[28:-28,28:-28] * mask_array[i])
    label.append(nib.load(w3[i]).get_data()[28:-28,28:-28] * mask_array[i])
    label.append(nib.load(t1[i]).get_data()[28:-28,28:-28] * mask_array[i])
    label.append(nib.load(t2[i]).get_data()[28:-28,28:-28] * mask_array[i])
    label.append(nib.load(t3[i]).get_data()[28:-28,28:-28] * mask_array[i]) 
    labels.append(label)
labels = np.array(labels)
brains = np.array(brains)
mask_array = np.array(mask_array)
lesion_mask_array = np.array(lesion_mask_array)


# In[8]:


len(labels)


# In[9]:


for number in numbers: #149
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


# In[10]:


mask_array = []
lesion_mask_array = []
brains = []
labels = []

for i in range(0,len(masks)):
    mask_array.append(nib.load(masks[i]).get_data()[28:-28,28:-28]) #[50:200,40:210]

for i in range(0,len(masks)):
    lesion_mask_array.append(np.clip(nib.load(lesion_masks[i]).get_data()[28:-28,28:-28], 0, 1))

for i in range(0,len(masks)):
    brain = []
    for j in range(6):
        brain.append(nib.load(fname[i]).get_data()[28:-28,28:-28,:,j]  * mask_array[i])
    brains.append(brain)
        
for i in range(0,len(masks)):
    label = []
    label.append(nib.load(w1[i]).get_data()[28:-28,28:-28] * mask_array[i])
    label.append(nib.load(w2[i]).get_data()[28:-28,28:-28] * mask_array[i])
    label.append(nib.load(w3[i]).get_data()[28:-28,28:-28] * mask_array[i])
    label.append(nib.load(t1[i]).get_data()[28:-28,28:-28] * mask_array[i])
    label.append(nib.load(t2[i]).get_data()[28:-28,28:-28] * mask_array[i])
    label.append(nib.load(t3[i]).get_data()[28:-28,28:-28] * mask_array[i]) 
    labels.append(label)
labels = np.array(labels)
brains = np.array(brains)
mask_array = np.array(mask_array)
lesion_mask_array = np.array(lesion_mask_array)


# count = 0
# for number in numbers: #149
#     file_root = "/data/Jeremy/Jeremy/" + number + "/"
#     label = labels[count]
#     mask = mask_array[count]
#     w1.append(file_root + "w1.nii.gz")
#     w2.append(file_root + "w2.nii.gz")
#     w3.append(file_root + "w3.nii.gz")
#     t1.append(file_root + "t1.nii.gz")
#     t2.append(file_root + "t2.nii.gz")
#     t3.append(file_root + "t3.nii.gz")
#     guess_sum_label = label[0] + label[1] + label[2] + 1e-16
#     label[0] /= guess_sum_label
#     label[1] /= guess_sum_label
#     label[2] /= guess_sum_label
#     label[0] *= 100
#     label[1] *= 100
#     label[2] *= 100  
#     label = np.nan_to_num(label)
#     label = np.clip(label, 0, 3000)
#     big_label = np.zeros((256,256,32))
#     big_label[28:-28,28:-28] = label[0] * mask
#     save_nii(big_label, file_root+"mwf.nii", w1[count])
#     count+=1

# In[22]:


def calculate_stats(idx):
    mean = labels[:,idx].mean()
    std = labels[:,idx].std()
    return mean, std


# In[ ]:


for i in range(len(labels)):
    labels[:,1][labels[:,0] > 20] = 0
    labels[:,2][labels[:,0] > 20] = 0
    labels[:,3][labels[:,0] > 20] = 0
    labels[:,4][labels[:,0] > 20] = 0
    labels[:,5][labels[:,0] > 20] = 0
    labels[:,0][labels[:,0] > 20] = 0


# In[ ]:


mean_0, std_0 = calculate_stats(0)
mean_1, std_1 = calculate_stats(1)
mean_2, std_2 = calculate_stats(2)
mean_3, std_3 = calculate_stats(3)
mean_4, std_4 = calculate_stats(4)
mean_5, std_5 = calculate_stats(5)


# In[ ]:


print(mean_0)
print(mean_1)
print(mean_2)
print(mean_3)
print(mean_4)
print(mean_5)
print("=")
print(std_0)
print(std_1)
print(std_2)
print(std_3)
print(std_4)
print(std_5)


# In[ ]:


from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import loggers
from Hang.unet3dPersonalGroupNorm_pytorchLightning import unet3d


# ### Testing

# #### Results

# In[16]:


import pandas as pd


# In[17]:


results = {}
results[0] = []
results[1] = []
results[2] = []
results[3] = []
results[4] = []
results[5] = []


# In[25]:


def calculateResultsErrorTable(state_dicts, idx=-15, save=False, make_frac = False, lesion=False):
    errs = {}
    errs[0] = []
    errs[1] = []
    errs[2] = []
    errs[3] = []
    errs[4] = []
    errs[5] = []
    
    for state_dict in state_dicts:
        model = unet3d(0.01,0.0012,is_deconv = False, weight=50).float()
        state_dict = torch.load(state_dict)
        model.load_state_dict(state_dict["state_dict"])
        model = model.float().cuda()
        model = model.eval()
        results = {}
        results[0] = []
        results[1] = []
        results[2] = []
        results[3] = []
        results[4] = []
        results[5] = []
        
        for idx in range(len(brains)):
            output = model(torch.tensor(np.divide(brains[idx], brains[idx][0] + 1e-16)[None]).cuda().float())
            label = labels[idx].copy()
            output[0,0] *= std_0
            output[0,0] += mean_0
            output[0,1] *= std_1
            output[0,1] += mean_1
            output[0,2] *= std_2
            output[0,2] += mean_2
            output[0,3] *= std_3
            output[0,3] += mean_3
            output[0,4] *= std_4
            output[0,4] += mean_4
            output[0,5] *= std_5
            output[0,5] += mean_5
            output = output.detach().cpu().numpy() 
            
            if (make_frac):
                guess_sum = output[0,0]+output[0,1]+output[0,2]
                output[0,0] /= guess_sum
                output[0,1] /= guess_sum
                output[0,2] /= guess_sum
                output[0,0] *= 100
                output[0,1] *= 100
                output[0,2] *= 100
                guess_sum_label = label[0] + label[1] + label[2] + 1e-16
                label[0] /= guess_sum_label
                label[1] /= guess_sum_label
                label[2] /= guess_sum_label
                label[0] *= 100
                label[1] *= 100
                label[2] *= 100  
            
            label = np.nan_to_num(label)
            mask = mask_array[idx]
            mask[output[0,0] > 60] = 0
            mask[label[0] > 60] = 0
            label = label * mask
            output = np.clip(output[0], 0, 3000)
            for feature_num in range(6):
                brain = np.ones((256,256,32), dtype=float)
                label_big = np.ones((256,256,32), dtype=float)
                if (lesion):
                    brain[28:-28,28:-28] = output[feature_num] * lesion_mask_array[idx]
                    label_big[28:-28,28:-28] = np.clip(label[feature_num] * lesion_mask_array[idx], 0, 3000)
                    if (np.sum(lesion_mask_array[idx]) == 0):
                        continue
                    results[feature_num].append(np.sum((-label_big+brain))/(np.sum(lesion_mask_array[idx])))
                else:
                    brain[28:-28,28:-28] = output[feature_num]*mask
                    label_big[28:-28,28:-28] =  np.clip(label[feature_num]*mask, 0, 3000)
                    results[feature_num].append(np.sum((-label_big+brain))/(np.sum(mask)))
        for feature_num in range(6):
            error = ""
            error += ("{:.3f}".format(np.array(results[feature_num]).mean()))
            error += ("±"+ "{:.3f}".format(np.array(results[feature_num]).std()))
            errs[feature_num].append(error)
    return errs


# ### This one below is the baseline

# In[15]:


errs = []
for i in range(6):
    state_dicts = ["/data/Jeremy/Jeremy/unetpp_smallds/baseline.ckpt",
                   "/data/Jeremy/Jeremy/unetpp_smallds/50_weight.ckpt", 
                   "/data/Jeremy/Jeremy/unetpp_smallds/100_weight.ckpt", 
                   "/data/Jeremy/Jeremy/unetpp_smallds/150_weight.ckpt", 
                   "/data/Jeremy/Jeremy/unetpp_smallds/200_weight.ckpt", 
                   "/data/Jeremy/Jeremy/unetpp_smallds_gpu2/250_weight.ckpt", 
                   "/data/Jeremy/Jeremy/unetpp_smallds_gpu2/275_weight.ckpt", 
                   "/data/Jeremy/Jeremy/unetpp_smallds_gpu2/300_weight.ckpt", 
                   "/data/Jeremy/Jeremy/unetpp_smallds_gpu2/325_weight.ckpt", 
                   "/data/Jeremy/Jeremy/unetpp_smallds_gpu2/350_weight.ckpt"]
    for state_dict in state_dicts:
        print(state_dict)
        print("feature: " + str(i))
        errs.append(calculateResultsError([state_dict], i, save=False, make_frac=True))
        errs.append(calculateResultsError([state_dict], i, save=False, make_frac=True, lesion=True))


# In[61]:


errs = {}
# for i in range(6):
for i in range(10):
    state_dicts = ["/data/Jeremy/Jeremy/unetpp_big/baseline.ckpt",
                   "/data/Jeremy/Jeremy/unetpp_big/50_weight.ckpt", 
                   "/data/Jeremy/Jeremy/unetpp_big/100_weight.ckpt", 
                   "/data/Jeremy/Jeremy/unetpp_big_gpu4/150_weight.ckpt", 
                   "/data/Jeremy/Jeremy/unetpp_big_gpu2/200_weight.ckpt", 
                   "/data/Jeremy/Jeremy/unetpp_big_gpu2/250_weight.ckpt", 
                   "/data/Jeremy/Jeremy/unetpp_big_gpu2/275_weight.ckpt", 
                   "/data/Jeremy/Jeremy/unetpp_big_gpu2/300_weight.ckpt", 
                   "/data/Jeremy/Jeremy/unetpp_big_gpu4/325_weight.ckpt"] #missing 350
    errs[i] = calculateResultsErrorTable(state_dicts, save=False, lesion = True, make_frac=True)
#     for state_dict in state_dicts:
#         print(state_dict)
#         print("feature: " + str(i))
#         errs.append(calculateResultsErrorTable([state_dict], i, save=False, make_frac=True))
#         errs.append(calculateResultsErrorTable([state_dict], i, save=False, make_frac=True, lesion=True))


# In[63]:


df = pd.DataFrame({
    "MWF": errs[0][0],
    "IEWF": errs[0][1],
    "CSFF": errs[0][2],
    "MWF T2": errs[0][3],
    "IEWF T2": errs[0][4],
    "CSFF T2": errs[0][5]
}, index=["Baseline", "50 Weight", "100 Weight", "150 Weight", "200 Weight", 
          "250 Weight", "275 Weight", "300 Weight", "325 Weight"]) #, "350 Weight"


# In[64]:


df.to_csv("lesion_50.csv")


# In[19]:


labels[0,0].mean()


# In[20]:


errs = calculateResultsError(["/data/Jeremy/Jeremy/unetpp_big/50_weight.ckpt"], i, save=False, make_frac=True, 
                             lesion=True)
# "/data/Jeremy/Jeremy/unetpp_big/baseline.ckpt",
# "/data/Jeremy/Jeremy/unetpp_big/50_weight.ckpt"


# In[18]:


errs = []
for i in range(6):
    state_dicts = ["/data/Jeremy/Jeremy/unetpp_big/100_weight.ckpt"]
    for state_dict in state_dicts:
        print(state_dict)
        print("feature: " + str(i))
        errs.append(calculateResultsError([state_dict], i, save=True, make_frac=True))
#         errs.append(calculateResultsError([state_dict], i, save=False, make_frac=True, lesion=True))


# In[18]:


errs = []
for i in range(0,1):
    state_dicts = ["/data/Jeremy/Jeremy/unetpp_big_gpu4/50_weight_3x3x3.ckpt"]
    for state_dict in state_dicts:
        print(state_dict)
        print("feature: " + str(i))
        errs.append(calculateResultsError([state_dict], i, save=True, make_frac=True))
        errs.append(calculateResultsError([state_dict], i, save=False, make_frac=True, lesion=True))


# In[18]:


errs = []
for i in range(0,1):
    state_dicts = ["/data/Jeremy/Jeremy/unetpp_big_gpu4/325_weight.ckpt"]
    for state_dict in state_dicts:
        print(state_dict)
        print("feature: " + str(i))
        errs.append(calculateResultsError([state_dict], i, save=True, make_frac=True))
        errs.append(calculateResultsError([state_dict], i, save=False, make_frac=True, lesion=True))


# In[89]:


errs = []
for i in range(0,1):
    state_dicts = ["/data/Jeremy/Jeremy/unetpp_big/50_weight.ckpt"]
    for state_dict in state_dicts:
        print(state_dict)
        print("feature: " + str(i))
        errs.append(calculateResultsError([state_dict], i, save=False, make_frac=True))
#         errs.append(calculateResultsError([state_dict], i, save=False, make_frac=True, lesion=True))


# In[105]:


full_errs_baseline = abs(np.array(errs_baseline[0][0]))
fileNamesFull_baseline = ""
for index in np.argsort(full_errs_baseline)[:40]:
    fileNamesFull_baseline += (numbers[index] + " ")
print(fileNamesFull_baseline)


# In[106]:


full_errs = abs(np.array(errs[0][0]))
fileNamesFull = ""
for index in np.argsort(full_errs)[:40]:
    fileNamesFull += (numbers[index] + " ")
print(fileNamesFull)


# In[128]:


goodBrains_baseline = fileNamesFull_baseline.split(" ")
goodBrains = fileNamesFull.split(" ")
goodBrainsList = ""
for name in goodBrains:
    if (name in goodBrains_baseline):
        goodBrainsList += (name + " ")
print(goodBrainsList)


# In[131]:


list = "2100 2049 1416 2115 1961 1531 1347 1757 2245 1892 2179 1876 0535 0623 0675 0668 0794 2128 2221 2161 1602 1743 1621 1749 1172 0282 1898 0131 1899 1735 0504 0857 1952"
len(list.split(" "))


# ### This one below is 100

# In[58]:


state_dicts = ["/data/Jeremy/Jeremy/unetpp_big/100_weight_full_brain.ckpt"]
errs.append(calculateResultsError(state_dicts, feature_num, save=True, make_frac=True))


# ### This one below is 150

# In[18]:


def calculateResultsError(state_dicts, feature_num, idx=-15, save=False, make_frac = False, lesion=False):
    for state_dict in state_dicts:
        weight = state_dict[idx:]
        model = unet3d(0.01,0.0012,is_deconv = False, weight=50).float()
        state_dict = torch.load(state_dict)
        model.load_state_dict(state_dict["state_dict"])
        model = model.float().cuda()
        model = model.eval()
        err = []
        rmse = []
        start = time.time()
      
        for idx in range(len(brains)):
            output = model(torch.tensor(np.divide(brains[idx], brains[idx][0] + 1e-16)[None]).cuda().float())
            label = labels[idx].copy()
            output[0,0] *= std_0
            output[0,0] += mean_0
            output[0,1] *= std_1
            output[0,1] += mean_1
            output[0,2] *= std_2
            output[0,2] += mean_2
            output[0,3] *= std_3
            output[0,3] += mean_3
            output[0,4] *= std_4
            output[0,4] += mean_4
            output[0,5] *= std_5
            output[0,5] += mean_5

            output = output.detach().cpu().numpy() 
            
            if (make_frac):
                guess_sum = output[0,0]+output[0,1]+output[0,2]
                output[0,0] /= guess_sum
                output[0,1] /= guess_sum
                output[0,2] /= guess_sum
                output[0,0] *= 100
                output[0,1] *= 100
                output[0,2] *= 100
                
                guess_sum_label = label[0] + label[1] + label[2] + 1e-16
                label[0] /= guess_sum_label
                label[1] /= guess_sum_label
                label[2] /= guess_sum_label
                label[0] *= 100
                label[1] *= 100
                label[2] *= 100
                
            
            
            label = np.nan_to_num(label)
            
            mask = mask_array[idx]
            mask[output[0,0] > 60] = 0
            mask[label[0] > 60] = 0
            label = label * mask

            brain = np.ones((256,256,32), dtype=float)
            output = np.clip(output[0,feature_num], 0, 3000)
            brain = output * mask
            label = label[feature_num]
            label = np.clip(label, 0, 3000)

            brain = np.ones((256,256,32), dtype=float)
            label_big = np.ones((256,256,32), dtype=float)
            if (lesion):
                brain = output * lesion_mask_array[idx]
                label_big = label * lesion_mask_array[idx]
                if (np.sum(lesion_mask_array[idx]) == 0):
                    continue
                rmse.append(np.sqrt(np.sum((-label_big+brain)**2)/np.sum(lesion_mask_array[idx])))
                err.append(np.sum((-label_big+brain))/(np.sum(lesion_mask_array[idx])))
            else:
                if (output.shape[0] < 256):
                    brain[28:-28,28:-28] = output*mask
                    label_big[28:-28,28:-28] = label*mask
                    rmse.append(np.sqrt(np.sum((-label_big+brain)**2)/np.sum(mask)))
                    err.append(np.sum((-label_big+brain))/(np.sum(mask)))
                    mask_big = np.ones((256,256,32), dtype=float)
                    mask_big[28:-28,28:-28] = mask
                else:
                    brain = output * mask
                    label_big = label * mask
                    rmse.append(np.sqrt(np.sum((-label_big+brain)**2)/np.sum(mask)))
                    err.append(np.sum((-label_big+brain))/(np.sum(mask)))

            
            if (feature_num == 0):
                path = w1[idx]
                subpath = "/mwf/"
            elif (feature_num == 1):
                path = w2[idx]
                subpath = "/iewf/"
            elif (feature_num == 2):
                path = w3[idx]
                subpath = "/csf/"
            elif (feature_num == 3):
                path = t1[idx]
                subpath = "/t1/"
            elif (feature_num == 4):
                path = t2[idx]
                subpath = "/t2/"
            elif (feature_num == 5):
                path = t3[idx]
                subpath = "/t3/"
            
            if (save):
                save_nii(brain, "results"+subpath+path[20:24]+"_mwf_"+str(feature_num)+"3x3x1.nii", path)
                save_nii(label_big, "results"+subpath+path[20:24]+"_label_"+str(feature_num)+".nii", path)
        end = time.time()
        print(end-start)
        print("err mean: " + str(np.array(err).mean()))
        print("err std: " + str(np.array(err).std()))
        print("rmse mean: " + str(np.array(rmse).mean()))
        print("rmse std: " + str(np.array(rmse).std()))
        return [err, rmse]


# In[ ]:




