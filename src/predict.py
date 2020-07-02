from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import Function

import torchvision.transforms as transforms
import models

from utils import weights_init, weights_init2

import logger as vanillaLog

import numpy as np
import pickle
import sys
import os
import matplotlib.pyplot as plt

import time

import gc
import argparse


# Get Python version. 
if sys.version.split(' ')[0][0] == '2':
    __python_version__      = 2
    text_input              = raw_input
elif sys.version.split(' ')[0][0] == '3':
    __python_version__      = 3
    text_input              = input


# Get Pytorch version.
__pytorch_version__     = int(torch.__version__.split('.')[1])
if __pytorch_version__ == 2:
    V                   = lambda x: x.data[0]
elif __pytorch_version__ == 4:
    V                   = lambda x: x.item()

# Argument parsing. The help keyword argument is descriptive. 
parser = argparse.ArgumentParser()
parser.add_argument('--test_dir_prefix', type=str, default='', help='Location of test images.')
parser.add_argument('--test_ims', type=str, default='', help='File specifying test images.')
parser.add_argument('--f_ext', type=str, default='.npy', help='Extension of files which store the deformation grids.')
parser.add_argument('--cuda', type=int, default=1, help='Whether to use CUDA.')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--load_model', default='', help="path to model (to continue training)")
# Parse arguments. 

args                            = parser.parse_args()

# Sanity checks. 
if args.load_model == '':
    print('Please specify load_model!')
    exit()


with open(os.path.join(args.load_model, 'opt.pkl'), 'rb') as fp:
    opt                         = pickle.load(fp)

# Whether to do validation and testing. 
do_testing                      = (args.test_dir_prefix != '')
args.do_testing                  = do_testing

if do_testing and not os.path.exists(args.test_dir_prefix):
    print('Specified directory for testing does not exist!')
    exit()

if do_testing:
    if args.test_dir_prefix[-1] != '/':
        args.test_dir_prefix    += '/'
    test_db_name                = os.path.dirname(args.test_dir_prefix).split('/')[-1]

# Create names of output directories. 
timestamp                       = time.strftime('%Y%m%d-%H%M%S') 
args.base_output_dir             = os.path.join('results/', '%s_%s_PRED_classifier_bs_%d_ds_%d_nc_%d_nhid_%s_lr_%g_optim_%s' %(timestamp, test_db_name, opt.batch_size, opt.ds_factor, opt.nc, '_'.join([str(s) for s in opt.n_hid]), opt.lr, opt.optim))

# If we don't want a hidden layer. 
if opt.n_hid == [-1]:
    opt.n_hid = []

test_imgs_dir, test_ims_filename    = os.path.split(args.test_ims)

# Print opt and ask to continue
opt_params_to_print             = [f for f in dir(args) if not f.startswith('_')]

print()
print('Configuration')
print('=============')
for param in opt_params_to_print:
    if not param.startswith('__'):
        sys.stdout.write('%30s:  ' %(param),)
        param_value             = eval('args.%s' %(param))
        if isinstance(param_value, int):
            sys.stdout.write('%d\n' %(param_value))
        elif isinstance(param_value, str):
            sys.stdout.write('%s\n' %(param_value))
        elif isinstance(param_value, float):
            sys.stdout.write('%g\n' %(param_value))
        elif isinstance(param_value, list):
            sys.stdout.write('[' + ', '.join([str(s) for s in param_value]) + ']\n')
        elif isinstance(param_value, bool):
            sys.stdout.write('%r\n' %(param_value))
print()
print('=============')

response = 'c'
while response not in ['y', 'Y', 'n', 'N', '']:
    response            = text_input('Continue (Y/n)? ')    
if response in ['n', 'N']:
    exit()

# Create output and logging directories. 
for directory in [args.base_output_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Create models. 
classifier                      = models.DiseaseClassifier(opt)
if opt.cuda:
    classifier                  = classifier.cuda()
print(classifier)
print('Attempting to load from %s.' %(args.load_model))
try:
    classifier.load_state_dict(torch.load(os.path.join(args.load_model, 'checkpoints/best_model/classifier.pth')))
except Exception as e: 
    print('Error loading previous model -- %s' %(e))
    exit()


# Create training criterion. 
BCECriterion                    = nn.BCEWithLogitsLoss() 
if opt.cuda:
    BCECriterion                = BCECriterion.cuda()

if do_testing:  
    test_dataset                = models.TestDatasetFold(args.test_dir_prefix, args.test_ims)


best_val_loss   = 1e5

# Save opt.
# Reset iter_mark to zero. 
iter_mark                       = 0

# Training batches are done. Now execute validation, if needed. 
if not do_testing:
    exit()

sys.stdout.write('Testing ...')
sys.stdout.flush()
# Validation has been demanded. 
test_dataloader             = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=int(args.workers))

# Create predictions array. 
predictions                 = np.zeros(len(test_dataset))
p_names                     = []

# Total val loss
total_test_loss             = 0.0

# Place the classifier in eval mode. 
classifier.eval()

# Store an ID, indicating till what step we have filled. 
filled_id                   = 0

# A sigmoid layer. 
sigmoid                     = nn.Sigmoid()

for batch_idx, data_point in enumerate(test_dataloader):
    # Collect garbage. 
    gc.collect()

    X, c                    = data_point

    # This batch size. Could differ from opt.batch_size for the last epoch. 
    this_batch_size         = X.shape[0]
    # If there is only one channel, we unsqueeze 1.
    if opt.nc == 1:
        X                   = X.unsqueeze(1)

    # Put on GPU if needed. 
    if opt.cuda:
        X                   = X.cuda()
    # Convert to Variable. 
    X                       = Variable(X)  

    # Run classifier. 
    y_hat                   = classifier(X)
    # Need to pass this through a Sigmoid, because classifier does not do it. 
    # y_hat                   = sigmoid(y_hat)

    # Remove y_hat from GPU if needed.
    if opt.cuda:
        y_hat               = y_hat.cpu()

    # Place predictions
    p_names                += list(c)
    predictions[filled_id:filled_id+this_batch_size]    = F.sigmoid(y_hat).data.numpy().squeeze() #F.softmax(y_hat, dim=1)[:,1].data.numpy().squeeze()

    filled_id               = filled_id+this_batch_size

    sys.stdout.write('.')
    sys.stdout.flush()

print()
with open(os.path.join(args.base_output_dir, 'predictions_%s_%s.txt' %(test_db_name, test_ims_filename)), 'w') as f_ptr:
    for name, score in zip(p_names, predictions):
        f_ptr.write('%s %.4f\n' %(name, score))
print('Predictions on test set saved to %s' %(os.path.join(args.base_output_dir, 'predictions_%s_%s.txt' %(test_db_name, test_ims_filename))))
print('Finished.')
print('=========================')

