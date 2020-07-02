import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.utils.data as data

import numpy as np

import sys
import os

# stats for jacobian_masked_resampled_resz_3o3
MEAN    = 0.0073717161525873615
STD     = 0.1050129681297089

# stats for jacobian_resampled_resz_3o3
MEAN    = 0.010051157482923154
STD     = 0.1218260979919173

# state for data_resz_3o3
MIN     = -2.702721
MAX     = 2.022158

GLOBAL_NORMALISE = True
ZERO_ONE_NORMALISE = False
SELF_ZERO_ONE_NORMALIZE = False

assert(
        ((not GLOBAL_NORMALISE) and (not ZERO_ONE_NORMALISE) and (SELF_ZERO_ONE_NORMALIZE)) or
        ((GLOBAL_NORMALISE) and (not ZERO_ONE_NORMALISE) and (not SELF_ZERO_ONE_NORMALIZE)) or
        ((not GLOBAL_NORMALISE) and (ZERO_ONE_NORMALISE) and (not SELF_ZERO_ONE_NORMALIZE))
)
        
class DiseaseClassifier(nn.Module):
    def __init__(self, opt):
        super(DiseaseClassifier, self).__init__()

        # List of hidden units per layer. 
        # Length of this list tells us the number of hidden layers. 
        self.n_hid          = opt.n_hid
        self.n_layers       = len(opt.n_hid)
        # Number of channels in input
        self.nc             = opt.nc
        # Size of the image. 
        self.img_size       = opt.img_size
        # Downsampling factor. 
        self.ds_factor      = opt.ds_factor

        conv_layers         = []
        linear_layers       = []

        ndf                 = 4
        prev_ndf            = self.nc

        # Introduce a convolutional kernel for each downsampling_factor. 
        for f in range(self.ds_factor):
            # Add a conv kernel. 
            conv_layers.append(nn.Conv3d(self.nc, self.nc, 3, stride=2, padding=1, bias=True))
            if f > -1:
                conv_layers.append(nn.BatchNorm3d(self.nc))
            # Add ReLU activation. 
            conv_layers.append(nn.ReLU(True))

        if len(conv_layers) > 0:
            self.conv       = nn.Sequential(*conv_layers)
        else:
            self.conv       = None

        # Visible units are now reduced by a factor determined by self.ds_factor. 
        prev_img_size       = self.img_size
        for u in range(self.ds_factor):
            next_img_size   = [int(np.ceil(u/2.0)) for u in prev_img_size]
            prev_img_size   = next_img_size
        self.n_vis          = int(np.prod(prev_img_size))*prev_ndf
        # Number of input units to the next linear layer. 
        prv_in_units        = self.n_vis

        for i, h in enumerate(self.n_hid):
            # Add a linear layer. 
            linear_layers.append(nn.Linear(prv_in_units, h))
            # Add a batch norm layer
            linear_layers.append(nn.BatchNorm1d(h))
            # Add a sigmoid layer. 
            linear_layers.append(nn.ReLU(True))
            # Add a dropout layer. 
            prv_in_units    = h

        # Finally, add another linear layer which gives 1 output. 
        # We don't add nn.Sigmoid() here because we will use nn.BCEWithLogitsLoss
        linear_layers.append(nn.Linear(prv_in_units, 1))
#        linear_layers.append(nn.Sigmoid())

        # This is the main. 
        self.linear         = nn.Sequential(*linear_layers)

    def forward(self, inputs):
        # Run inputs on conv first, if conv is not None. 
        if self.conv is not None:
            x               = self.conv(inputs)
            # Flatten for linear units. 
            x               = x.view(-1, self.n_vis)
            return self.linear(x)
        else:
            # Simply apply the linear units. 
            return self.linear(inputs.view(-1, self.n_vis))

      
# Dataset we will use to load training images. 
# This also loads the disease label with the image. 
class TrainDataset(data.Dataset):
    def __init__(self, dir_path, f_ext):
        super(TrainDataset, self).__init__()

        # Save dir path. 
        self.dir_path       = dir_path

        # Get the list of files in dir_path.
        file_list           = os.listdir(dir_path)
        # Filter files from this list, keeping only those with extension f_ext. 
        file_list           = [t for t in file_list if t.endswith(f_ext)]

        # Record this as imgs
        self.imgs           = file_list

        # Labels are given as the first character of the file name. 
        self.labels         = [int(f.split('_')[0]) for f in self.imgs]
        self.names          = [f.split('_')[1:] for f in self.imgs]

    def __getitem__(self, index):
        # Read the image at position index. 
        # img                 = (np.load(os.path.join(self.dir_path, self.imgs[index])) - self.mean)/self.std
        img                 = np.load(os.path.join(self.dir_path, self.imgs[index]))# /255.0
        # Normalisation
        if GLOBAL_NORMALISE:
            img             = (img - MEAN)# / STD        
        elif ZERO_ONE_NORMALISE:
            img             = (img - MIN)/(MAX - MIN)
        elif SELF_ZERO_ONE_NORMALIZE:
            img             = (img - img.min())/(img.max() - img.min())

        label               = np.float32(self.labels[index])
        name                = self.names[index]
        return img, label, name

    def __len__(self):
        return len(self.imgs)

# Dataset to load test images. 
# This does not load the image label with the images. 
class TestDataset(data.Dataset):
    def __init__(self, dir_path, f_ext):
        super(TestDataset, self).__init__()

        # Save dir path. 
        self.dir_path       = dir_path

        # Get the list of files in dir_path.
        file_list           = os.listdir(dir_path)
        # Filter files from this list, keeping only those with extension f_ext. 
        file_list           = [t for t in file_list if t.endswith(f_ext)]

        self.imgs           = file_list

    def __getitem__(self, index):
        # img                 = (np.load(os.path.join(self.dir_path, self.imgs[index])) - self.mean)/self.std
        img                 = np.load(os.path.join(self.dir_path, self.imgs[index]))#/255.0
        # Normalisation
        if GLOBAL_NORMALISE:
            img             = (img - MEAN) #/ STD
        elif ZERO_ONE_NORMALISE:
            img             = (img - MIN)/(MAX - MIN)
        elif SELF_ZERO_ONE_NORMALIZE:
            img             = (img - img.min())/(img.max() - img.min())
        name                = self.imgs[index]
        return img, name

    def __len__(self):
        return len(self.imgs)

# TrainDatasetFold, which takes as input a .txt file specifying a training fold.
class TrainDatasetFold(data.Dataset):
    def __init__(self, dir_path, f_fold):
        super(TrainDatasetFold, self).__init__()

        # Save dir path. 
        self.dir_path       = dir_path

        # Read image list. 
        self.imgs           = np.loadtxt(f_fold, dtype=str)


        # Make sure that all files specified in f_fold are present in self.dir_path. 
        for f_name in self.imgs:
            if not os.path.exists(os.path.join(self.dir_path, f_name)):
                print('%s does not exist!' %(os.path.join(self.dir_path, f_name)))
                exit()

        # Labels are given as the first character of the file name. 
        self.labels         = [int(f.split('_')[0]) for f in self.imgs]
        self.names          = [f.split('_')[1:] for f in self.imgs]
    
    def __getitem__(self, index):
        # img                 = (np.load(os.path.join(self.dir_path, self.imgs[index])) - self.mean)/self.std
        img                 = np.load(os.path.join(self.dir_path, self.imgs[index]))#/255.0
        # Normalisation
        if GLOBAL_NORMALISE:
            img             = (img - MEAN) #/ STD
        elif ZERO_ONE_NORMALISE:    
            img             = (img - MIN)/(MAX - MIN)
        elif SELF_ZERO_ONE_NORMALIZE:
            img             = (img - img.min())/(img.max() - img.min())

        label               = np.float32(self.labels[index])
        name                = self.names[index]

        return img, label, name

    def __len__(self):
        return len(self.imgs)


# TestDatasetFold, which takes as input a .txt file specifying a testing fold.
class TestDatasetFold(data.Dataset):
    def __init__(self, dir_path, f_fold):
        super(TestDatasetFold, self).__init__()

        # Save dir path. 
        self.dir_path       = dir_path

        # Read image list. 
        self.imgs           = np.loadtxt(f_fold, dtype=str)

        # Make sure that all files specified in f_fold are present in self.dir_path. 
        for f_name in self.imgs:
            if not os.path.exists(os.path.join(self.dir_path, f_name)):
                print('%s does not exist!' %(os.path.join(self.dir_path, f_name)))
                exit()
    
    def __getitem__(self, index):
        # img                 = (np.load(os.path.join(self.dir_path, self.imgs[index])) - self.mean)/self.std
        img                 = np.load(os.path.join(self.dir_path, self.imgs[index]))#/255.0
        # Normalisation
        if GLOBAL_NORMALISE:
            img             = (img - MEAN)# / STD    
        elif ZERO_ONE_NORMALISE:
            img             = (img - MIN)/(MAX - MIN)
        elif SELF_ZERO_ONE_NORMALIZE:
            img             = (img - img.min())/(img.max() - img.min())
        name                = self.imgs[index]

        return img, name

    def __len__(self):
        return len(self.imgs)

