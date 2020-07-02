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
parser.add_argument('--ds_factor', type=int, default=0, help='The downsampling factor. This number indicates the factor (which is a power of 2 == 2**ds_factor) by which we should downsample.')
parser.add_argument('--n_hid', type=int, nargs='+', default=[], help='List of hidden units. There is one number for each hidden layer.')
parser.add_argument('--img_size', type=int, nargs='+', default=[], help='Size of the jacobian tensors.')
parser.add_argument('--data_dir_prefix', type=str, default='', help='Location of the data directory')
parser.add_argument('--val_dir_prefix', type=str, default='', help='Location of validation images.')
parser.add_argument('--test_dir_prefix', type=str, default='', help='Location of test images.')
parser.add_argument('--f_ext', type=str, default='.npy', help='Extension of files which store the deformation grids.')
parser.add_argument('--cuda', type=int, default=1, help='Whether to use CUDA.')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batch_size', type=int, default=-1, help='input batch size')
parser.add_argument('--nc', type=int, default=3, help='Number of channels in input images.')
parser.add_argument('--n_epochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--reduce_lr_every', type=int, default=-1, help='Reduce learning rate every ___ epochs.')
parser.add_argument('--reduce_lr_factor', type=float, default=0.5, help='Reduce learning rate by this factor.')
parser.add_argument('--load_model', default='', help="path to model (to continue training)")
parser.add_argument('--resume_epoch', type=int, default=0, help='Epoch from which to resume training.')
parser.add_argument('--checkpoint_every', type=int, default=5, help='How frequently to checkpoint (epochs). A value of -1 says no checkpointing is to be done.')
parser.add_argument('--optim', type=str, default='Adam', help='Which optimiser to use.')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam OR momentum for SGD. default=0.5')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay parameter for SGD.')
parser.add_argument('--splits_dir_prefix', type=str, default='', help='Path to the directory containing splits.')
parser.add_argument('--split_id', type=int, default=0, help='Which cross validation split to use.')
# Parse arguments. 

opt                             = parser.parse_args()

if opt.img_size == []:
    print('Please specify a valid img size: --img_size DEPTH HEIGHT WIDTH')
    exit(1)

if opt.splits_dir_prefix != '':
    opt.val_dir_prefix          = opt.data_dir_prefix

# Sanity checks. 
if opt.load_model != '' and opt.resume_epoch == 0:
    print('Please specify epoch to resume if load_model is set!')
    exit()
if opt.batch_size <= 0:
    print('Please specify batch size!')
    exit()
if opt.optim not in ['Adam', 'SGD']:
    print('Acceptable values for --optim are [%s]' %(', '.join(['Adam', 'SGD'])))
    exit()

# Whether to do validation and testing. 
do_validation                   = (opt.val_dir_prefix != '')
do_testing                      = (opt.test_dir_prefix != '')
opt.do_validation               = do_validation
opt.do_testing                  = do_testing

if opt.data_dir_prefix == '':
    print('Please specify training data directory (--data_dir_prefix)!')
    exit()
if not os.path.exists(opt.data_dir_prefix):
    print('Specified data directory does not exist!')
    exit()
if do_validation and not os.path.exists(opt.val_dir_prefix):
    print('Specified directory for validation does not exist!')
    exit()
if do_testing and not os.path.exists(opt.test_dir_prefix):
    print('Specified directory for testing does not exist!')
    exit()

# Extract a prefix name from the data_dir_prefix. 
if opt.data_dir_prefix[-1] != '/':
    opt.data_dir_prefix        += '/'
train_db_name                   = os.path.dirname(opt.data_dir_prefix).split('/')[-1]
if opt.splits_dir_prefix != '':
    if opt.splits_dir_prefix[-1] != '/':
        opt.splits_dir_prefix  += '/'
    train_db_name               = 's'+str(opt.split_id) + '_' + os.path.dirname(opt.splits_dir_prefix).split('/')[-1] + '_' + train_db_name

if do_validation:
    if opt.val_dir_prefix[-1] != '/':
        opt.val_dir_prefix     += '/'
    val_db_name                 = os.path.dirname(opt.val_dir_prefix).split('/')[-1]

if do_testing:
    if opt.data_dir_prefix[-1] != '/':
        opt.test_dir_prefix    += '/'
    test_db_name                = os.path.dirname(opt.test_dir_prefix).split('/')[-1]

# Create names of output directories. 
timestamp                       = time.strftime('%Y%m%d-%H%M%S') 
opt.base_output_dir             = os.path.join('results/', '%s_%s_classifier_bs_%d_ds_%d_nc_%d_nhid_%s_lr_%g_optim_%s' %(timestamp, train_db_name, opt.batch_size, opt.ds_factor, opt.nc, '_'.join([str(s) for s in opt.n_hid]), opt.lr, opt.optim))
opt.checkpoints_dir             = os.path.join(opt.base_output_dir, 'checkpoints/')
opt.best_model_dir              = os.path.join(opt.checkpoints_dir, 'best_model/')
opt.logs_dir                    = os.path.join(opt.base_output_dir, 'vlogs/')
opt.train_log_file              = os.path.join(opt.logs_dir, 'train.csv')
opt.val_log_file                = os.path.join(opt.logs_dir, 'train_val.csv')

# Fix the image size. This was used earlier by Mihir. Please do not uncomment. 
#opt.img_size                    = [91, 57, 87]

# If we don't want a hidden layer. 
if opt.n_hid == [-1]:
    opt.n_hid = []


# Print opt and ask to continue
opt_params_to_print             = [f for f in dir(opt) if not f.startswith('_')]

print()
print('Configuration')
print('=============')
for param in opt_params_to_print:
    if not param.startswith('__'):
        sys.stdout.write('%30s:  ' %(param),)
        param_value             = eval('opt.%s' %(param))
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
for directory in [opt.base_output_dir, opt.checkpoints_dir, opt.logs_dir, opt.best_model_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Create models. 
classifier                      = models.DiseaseClassifier(opt)
if opt.cuda:
    classifier                  = classifier.cuda()
print(classifier)
for m in classifier.modules():
    weights_init(m, nonlinearity='relu')
# Create training criterion. 
BCECriterion                    = nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss() #weight=torch.FloatTensor([68.0/159.0, 91.0/159.0]))
if opt.cuda:
    BCECriterion                = BCECriterion.cuda()


# Initialise optimiser. 
if opt.optim == 'Adam':
    optimiser_classifier        = optim.Adam(classifier.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
elif opt.optim == 'SGD':
    optimiser_classifier        = optim.SGD(classifier.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, momentum=opt.beta1)

# Create datasets.
if opt.splits_dir_prefix == '':
    train_dataset               = models.TrainDataset(opt.data_dir_prefix, opt.f_ext)
else:
    train_dataset               = models.TrainDatasetFold(opt.data_dir_prefix, os.path.join(opt.splits_dir_prefix, '%d_train.txt' %(opt.split_id)))

if do_validation:
    if opt.splits_dir_prefix == '':
        val_dataset             = models.TrainDataset(opt.val_dir_prefix, opt.f_ext)
    else:
        val_dataset             = models.TrainDatasetFold(opt.data_dir_prefix, os.path.join(opt.splits_dir_prefix, '%d_val.txt' %(opt.split_id)))

if do_testing:  
    test_dataset                = models.TestDataset(opt.test_dir_prefix, opt.f_ext)
    test_dataset                = models.TestDatasetFold(opt.test_dir_prefix, os.path.join(opt.splits_dir_prefix, '%d_test.txt' %(opt.split_id)))


# Check if model to resume. 
resume_iter_mark = -1
if opt.load_model != '':
    print('Previous model specified. Attempting to load ...')
    try:
        classifier.load_state_dict(torch.load(os.path.join(opt.load_model, 'checkpoints/%d_classifier.pth' %(opt.resume_epoch))))
        optimiser_classifier.load_state_dict(torch.load(os.path.join(opt.load_model, 'checkpoints/%d_optimiser_classifier.pth' %(opt.resume_epoch))))
    except Exception as e: 
        print('Error loading previous model -- %s' %(e))
        exit()

    previous_log_file           = os.path.join('/'.join(opt.load_model.split('/')[:-1]), 'vlogs/train.csv')
    previous_error_rec          = np.loadtxt(previous_log_file, delimiter=',', dtype=str)
    resume_iter_mark            = previous_error_rec.shape[0]

    print('Successfully loaded previous model. I will attempt to resume from epoch %d of training (Iteration %d).' %(opt.resume_epoch, resume_iter_mark))
else:
    print('No model specified. Starting with a fresh model.')


best_val_loss   = 1e5

# Save opt.
with open(os.path.join(opt.base_output_dir, 'opt.pkl'), 'wb') as fptr_opt:
    opt_pickle  = pickle.dump(opt, fptr_opt, pickle.HIGHEST_PROTOCOL)

# Reset iter_mark to zero. 
iter_mark                       = 0

for epoch in range(opt.n_epochs):
    cur_epoch                   = opt.resume_epoch + epoch

    # Place model in train mode.
    classifier.train()

    # Create training batches. 
    train_dataloader            = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.workers))

    # Reset total train loss
    total_train_loss            = 0.0

    # Iterate over mini-batches. 
    for batch_idx, data_point in enumerate(train_dataloader):
        # Collect garbage. 
        gc.collect()

        X, y, c                 = data_point

        # This batch size. Could differ from opt.batch_size for the last batch. 
        this_batch_size         = X.shape[0]
        # If there is only one channel, we unsqueeze 1.
        if opt.nc == 1:
            X                   = X.unsqueeze(1)

        # Set to cuda. 
        if opt.cuda:
            X                   = X.cuda()
            y                   = y.cuda()
        # Convert to variables. 
        X                       = Variable(X)
        y                       = Variable(y.unsqueeze(1))

        # Reset gradients of optimiser. 
        optimiser_classifier.zero_grad()

        # Forward pass on the classifier. 
        y_hat                   = classifier(X)

        # Compute loss
        loss_bce                = BCECriterion(y_hat, y) #.squeeze().long())

        # Backward on the computed loss. 
        loss_bce.backward()

        # Update params. 
        optimiser_classifier.step()

        # Report statistics. 
        print('Epoch [%d], Iteration [%d]: loss_bce: %.4f' %(cur_epoch, iter_mark, V(loss_bce)))

        # Increment iter_mark. 
        iter_mark              += 1
        # Update total train loss. 
        total_train_loss       += V(loss_bce)*this_batch_size

        # Log information. 
        info                    = {
                'loss_bce':         V(loss_bce)
        }
        vanillaLog.log_info(opt.train_log_file, info, iter_mark)
    
    # Save models if checkpoint_every is satisfied. 
    if opt.checkpoint_every > 0 and (cur_epoch + 1)%opt.checkpoint_every == 0:
        torch.save(classifier.state_dict(), os.path.join(opt.checkpoints_dir, '%d_classifier.pth' %(cur_epoch)))
        torch.save(optimiser_classifier.state_dict(), os.path.join(opt.checkpoints_dir, '%d_optimiser_classifier.pth' %(cur_epoch)))
        print('Checkpoints saved at %s' %(opt.checkpoints_dir))

    # Whether we should reduce the learning rate. 
    if opt.reduce_lr_every > 0 and (cur_epoch + 1)%opt.reduce_lr_every  == 0:
         print('Reducing learning rate ...')
         for obj in [optimiser_classifier]:
             for param_group in obj.param_groups:
                 param_group['lr'] *= opt.reduce_lr_factor

    print()
    print('Train epoch %d. Train loss: %.4f' %(cur_epoch, total_train_loss/len(train_dataset)))
    print('=========================')
    # Training batches are done. Now execute validation, if needed. 
    if not do_validation: 
        continue
    
    sys.stdout.write('Validating ')
    sys.stdout.flush()
    # Validation has been demanded. 
    val_dataloader              = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=int(opt.workers), drop_last=True)

    # Total val loss
    total_val_loss              = 0.0

    # Place the classifier in eval mode. 
    classifier.eval()

    for batch_idx, data_point in enumerate(val_dataloader):
        # Collect garbage. 
        gc.collect()

        X, y, c                 = data_point

        # This batch size. Could differ from opt.batch_size for the last epoch. 
        this_batch_size         = X.shape[0]
        # If there is only one channel, we unsqueeze 1.
        if opt.nc == 1:
            X                   = X.unsqueeze(1)

        # Put on GPU if needed. 
        if opt.cuda:
            X                   = X.cuda()
            y                   = y.cuda()
        # Convert to Variable. 
        X                       = Variable(X)
        y                       = Variable(y.unsqueeze(1))

        # Run classifier. 
        y_hat                   = classifier(X)

        # Compute loss. 
        loss_bce                = BCECriterion(y_hat, y) #.squeeze().long())
        # Add to total val loss. 
        total_val_loss         += V(loss_bce)*this_batch_size

        sys.stdout.write('.')
        sys.stdout.flush()

    print('\n')
    print('Validation epoch %d. Validation loss: %.4f' %(cur_epoch, total_val_loss/len(val_dataset)))
    print('=========================')

    train_val_info              = {
        'train_loss':           total_train_loss/len(train_dataset),
        'val_loss':             total_val_loss/len(val_dataset)
    }
    vanillaLog.log_info(opt.val_log_file, train_val_info, epoch)

    # Save models if checkpoint_every is satisfied. 
    if (total_val_loss / len(val_dataset) ) < best_val_loss:
        best_val_loss = total_val_loss / len(val_dataset)
        torch.save(classifier.state_dict(), os.path.join(opt.best_model_dir, 'classifier.pth'))
        torch.save(optimiser_classifier.state_dict(), os.path.join(opt.best_model_dir, 'optimiser_classifier.pth'))
        print('Best model so far saved at %s' %(opt.best_model_dir))

# =============== End of train loop ===============

# Save models one final time. 
if opt.n_epochs > 0:
    sys.stdout.write('Saving models ...')
    sys.stdout.flush()
    torch.save(classifier.state_dict(), os.path.join(opt.checkpoints_dir, '%d_classifier.pth' %(cur_epoch)))
    torch.save(optimiser_classifier.state_dict(), os.path.join(opt.checkpoints_dir, '%d_optimiser_classifier.pth' %(cur_epoch)))
    print('done.')

# Load best model
classifier.load_state_dict(torch.load(os.path.join(opt.best_model_dir, 'classifier.pth')))
optimiser_classifier.load_state_dict(torch.load(os.path.join(opt.best_model_dir, 'optimiser_classifier.pth')))

# Training batches are done. Now execute validation, if needed. 
if not do_testing:
    exit()

sys.stdout.write('Testing ')
sys.stdout.flush()
# Validation has been demanded. 
test_dataloader             = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=int(opt.workers))

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
with open(os.path.join(opt.base_output_dir, 'predictions_%s.txt' %(test_db_name)), 'w') as f_ptr:
    for name, score in zip(p_names, predictions):
        f_ptr.write('%s %.4f\n' %(name, score))
print('Predictions on test set saved to %s' %(os.path.join(opt.base_output_dir, 'predictions_%s.txt' %(test_db_name))))
print('Finished.')
print('=========================')

