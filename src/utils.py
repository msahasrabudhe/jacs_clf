import torch
import torch.nn as nn
import sys
from torch.autograd import Variable

def write_flush(text, stream=sys.stdout):
    stream.write(text)
    stream.flush()
    
def parse_input(images):
    images = images.float()/255.0
    return images

def weights_init(m, nonlinearity='relu'):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nonlinearity)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def weights_init2(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_year(t):
    if '-' in t:
        return int(t.split('-')[-1])
    elif '/' in t:
        return int(t.split('/')[-1])


