import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', '..')))
from src.torchdiffeq import odeint_adjoint as odeint

class ODEfunc(nn.Module):
    """The main function to be used in ODE solver"""

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):
    """ the block within which the odefunction is optimized 
    NOTE: This code is adapted from the main files of https://github.com/rtqichen/torchdiffeq/
    for which all auxiliary functions can be found in torchdiffeq directory.
    """
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-5, atol=1e-5)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

class MNet(nn.Module):
    """ Initializes 4 downsampling networks for fft and signal sources at large and subsample dimensions
    Concatenates results of full and subsample downsampling for fft and signal.

     """
    def __init__(self):
        super(MNet, self).__init__()

        self.signet_full  = DownSampleSig_full()
        self.fftnet_full = DownSampleFFT_full()
        self.signet_sub  = DownSampleSig_sub()
        self.fftnet_sub = DownSampleFFT_sub()

    def forward(self, x, y):
        if x.size()[2]==750:
            x = self.signet_sub(x)
            y = self.fftnet_sub(y)
        else:
            x = self.signet_full(x)
            y = self.fftnet_full(y)
        xy = torch.cat((x,y),2)
        out = xy.view(-1,128,5,5)
        return out
    
class ODEModel(nn.Module):
    """ Initializes all networks for the odesolver.

     """
    def __init__(self):
        super(ODEModel, self).__init__()
        
        self.downsample = MNet()
        self.feature_layers = ODEBlock(ODEfunc(128))
        self.fc_layers = nn.Sequential(
            norm(128),
            nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(128, 32)
        )
        
    def forward(self,x,y):
        out = self.downsample(x,y)
        out = self.feature_layers(out)
        out = self.fc_layers(out)
        return out

class DownSampleFFT_sub(nn.Module):
    """Downsample FFT signal for 1/5th of the sample"""
    def __init__(self,conv1_num_filters=64, conv2_num_filters=128, \
        conv_ksize=16, conv_stride=1, conv_padding=2, \
        pool_ksize=32, pool_stride=1, pool_padding=2 ):
        super(DownSampleFFT_sub, self).__init__()
  
        # Set hyperparameters needed in forward
        self.conv1_num_filters, self.conv2_num_filters= conv1_num_filters, conv2_num_filters

        # Define layers
        #   Conv 1
        self.conv1 = nn.Conv1d(1, self.conv1_num_filters, \
                               kernel_size=conv_ksize*2, stride=conv_stride+2, padding=conv_padding)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv1_bn = nn.BatchNorm1d(self.conv1_num_filters)
        self.pool1 = nn.MaxPool1d(kernel_size=pool_ksize*2, stride=pool_stride+1, padding=pool_padding)

        #   Conv2
        self.conv2 = nn.Conv1d(self.conv1_num_filters, self.conv2_num_filters, \
                               kernel_size=conv_ksize, stride=conv_stride+1, padding=conv_padding)

        self.lr = nn.LeakyReLU(0.02)
        self.drop = nn.Dropout(0.15)
        
    def forward(self, x):

        x = self.lr(self.conv1_bn(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop(x)
        x = self.conv2(x)

        return x

class DownSampleSig_sub(nn.Module):
    """Downsample  signal for 1/5th of the sample"""

    def __init__(self,conv1_num_filters=128, conv_ksize=32, conv_stride=3, conv_padding=1, \
        pool_ksize=32, pool_stride=2, pool_padding=0):
        
        super(DownSampleSig_sub, self).__init__()
  
        # Set hyperparameters needed in forward
        self.conv1_num_filters = conv1_num_filters

        # Define layers
        #   Conv 1
        self.conv1 = nn.Conv1d(1, self.conv1_num_filters, \
                               kernel_size=conv_ksize*2, stride=conv_stride+1, padding=conv_padding,dilation=1)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv1_bn = nn.BatchNorm1d(self.conv1_num_filters)
        self.pool1 = nn.MaxPool1d(kernel_size=pool_ksize*2, stride=pool_stride+1, padding=pool_padding,dilation=2)

        
        #relu and dropout
        self.drop = nn.Dropout(0.15)
        self.lr = nn.LeakyReLU(0.02)     

    def forward(self, x):

        x = self.lr(self.conv1_bn(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop(x)

        return x
class DownSampleSig_full(nn.Module):
    """Downsample  full signal for  ODE solver"""

    def __init__(self,conv1_num_filters=16, conv2_num_filters=32, conv3_num_filters=128, \
        conv_ksize=32, conv_stride=1, conv_padding=1, \
        pool_ksize=32, pool_stride=1, pool_padding=0, \
        num_linear=128,p=0.5):
        super(DownSampleSig_full, self).__init__()
        
        # Set hyperparameters needed in forward
        self.conv1_num_filters, self.conv2_num_filters, self.conv3_num_filters= conv1_num_filters, conv2_num_filters,conv3_num_filters

        # Define layers
        #   Conv 1
        self.conv1 = nn.Conv1d(1, self.conv1_num_filters, \
                               kernel_size=conv_ksize*2, stride=conv_stride+1, padding=conv_padding,dilation=1)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv1_bn = nn.BatchNorm1d(self.conv1_num_filters)
        self.pool1 = nn.MaxPool1d(kernel_size=pool_ksize*2, stride=pool_stride+1, padding=pool_padding,dilation=2)

        #   Conv2
        self.conv2 = nn.Conv1d(self.conv1_num_filters, self.conv2_num_filters, \
                               kernel_size=conv_ksize*2, stride=conv_stride+1, padding=conv_padding,dilation=2)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm1d(self.conv2_num_filters)
        
        self.pool2 = nn.MaxPool1d(kernel_size=pool_ksize, stride=pool_stride+1, padding=pool_padding,dilation=2)

        # Conv3
        
        self.conv3 = nn.Conv1d(self.conv2_num_filters, self.conv3_num_filters, \
                                kernel_size=conv_ksize, stride=conv_stride+1, padding=conv_padding,dilation=4)
        
        #relu and dropout
        self.lr = nn.LeakyReLU(0.02)
        self.drop = nn.Dropout(0.15)

    def forward(self, x):

        x = self.lr(self.conv1_bn(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop(x)
        x = self.lr(self.conv2_bn(self.conv2(x)))
        x = self.pool2(x)
        x = self.drop(x)

        x = self.conv3(x)
        return x
    
class DownSampleFFT_full(nn.Module):
    """Downsample  fft for full signal for  ODE solver"""

    def __init__(self,conv1_num_filters=16, conv2_num_filters=32, conv3_num_filters=128, \
        conv_ksize=32, conv_stride=1, conv_padding=1, \
        pool_ksize=32, pool_stride=1, pool_padding=1):
        super(DownSampleFFT_full, self).__init__()
        self.lr = nn.LeakyReLU(0.02)     
  
        # Set hyperparameters needed in forward
        self.conv1_num_filters, self.conv2_num_filters, self.conv3_num_filters= conv1_num_filters, conv2_num_filters,conv3_num_filters

        # Define layers
        
        #   Conv 1
        self.conv1 = nn.Conv1d(1, self.conv1_num_filters, \
                               kernel_size=conv_ksize*2, stride=conv_stride+2, padding=conv_padding)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv1_bn = nn.BatchNorm1d(self.conv1_num_filters)
        self.pool1 = nn.MaxPool1d(kernel_size=pool_ksize*2, stride=pool_stride+1, padding=pool_padding)

        #   Conv2
        self.conv2 = nn.Conv1d(self.conv1_num_filters, self.conv2_num_filters, \
                               kernel_size=conv_ksize, stride=conv_stride+1, padding=conv_padding)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm1d(self.conv2_num_filters)
        
        self.pool2 = nn.MaxPool1d(kernel_size=pool_ksize, stride=pool_stride+1, padding=pool_padding)

        # Conv3
        
        self.conv3 = nn.Conv1d(self.conv2_num_filters, self.conv3_num_filters, \
                                kernel_size=conv_ksize, stride=conv_stride+1, padding=conv_padding)
        
        #leaky relu and dropout
        self.lr = nn.LeakyReLU(0.02)
        self.drop = nn.Dropout(0.15)        
    def forward(self, x):
        x = self.lr(self.conv1_bn(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop(x)
        x = self.lr(self.conv2_bn(self.conv2(x)))
        x = self.pool2(x)
        x = self.drop(x)
        x = self.conv3(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):

        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)



class ConcatConv1d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv1d, self).__init__()
        module = nn.ConvTranspose1d if transpose else nn.Conv1d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

