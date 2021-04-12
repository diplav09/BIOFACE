import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, kernel_size, stride, dilation, padding):
    return nn.Conv2d(in_channels, out_channels, kernel_size,stride, dilation, padding)

def conv(in_channels, out_channels, kernel_size, bias= True, padding = 1, stride = 1):
	return nn.Conv2d(in_channels, out_channels, kernel_size, padding= (kernel_size//2), bias = bias, stride = stride)


class SingleConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 1, dilation = 1, groups = 1, relu = True, bn = True, bias = False):
		super(SingleConv, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, groups = groups, bias = bias)
		self.bn = nn.BatchNorm2d(out_channels, eps = 1e-5, momentum = 0.01, affine = True) if bn else None
		self.relu = nn. ReLU() if relu else None

	def forward(self, x):
		x = self.conv(x)
		if self.bn is not None:
			x = self.bn(x)
		if self.relu is not None:
			x = self.relu(x)
		return x

class TripleConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 1, dilation = 1, groups = 1, relu = True, bias = False):
		super(TripleConv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, groups = groups, bias = bias),
			nn.BatchNorm2d(out_channels, eps = 1e-5, momentum = 0.01, affine = True),
			nn.ReLU(inplace = True),
			nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, groups = groups, bias = bias),
			nn.BatchNorm2d(out_channels, eps = 1e-5, momentum = 0.01, affine = True),
			nn.ReLU(inplace = True),
			nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, groups = groups, bias = bias),
			nn.BatchNorm2d(out_channels, eps = 1e-5, momentum = 0.01, affine = True),
			nn.ReLU(inplace = True)
		)
	def forward(self, x):
		x = self.conv(x)
		return x

class Encoder(nn.Module):
	def __init__(self,in_channels):
		super(Encoder, self).__init__()
		self.layer1 = TripleConv(in_channels,32,3)
		self.layer2 = TripleConv(32,64,3)
		self.layer3 = TripleConv(64,128,3)
		self.layer4 = TripleConv(128,256,3)
		self.layer5 = TripleConv(256,512,3)
		self.mp = nn.MaxPool2d(2,stride = 2)
	def forward(self, x):
		x1  = self.layer1(x)
		x1d = self.mp(x1)
		x2  = self.layer2(x1d)
		x2d = self.mp(x2)
		x3  = self.layer3(x2d)
		x3d = self.mp(x3)
		x4  = self.layer4(x3d)
		x4d = self.mp(x4)
		x5  = self.layer5(x4d)
		return x1,x2,x3,x4,x5

class Up(nn.Module):
	def _init__(self ,in_channels ,out_channels, bilinear = False):
		super(Up, self).__init__()
		if bilinear:
			self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corner = True)
		else:
			self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride = 2)
		self.conv = TripleConv(out_channels,out_channels, 3)
	def forward(self, x1, x2):
		x1 = self.up(x1)

		#to resolve padding issue
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]

		x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffy //2))

		#using add instead of concat to reduce complexity
		x = torch.add(x2,x1)
		x = self.conv(x)
		return x

class Decoder(nn.Module):
	def _init__(self,out_channels):
		super(Decoder, self).__init__()
		self.up1 = Up(512,256)
		self.up2 = Up(256,128)
		self.up3 = Up(128,64)
		self.up4 = Up(64,32)
		self.up5 = SingleConv(32, out_channels, 3)

	def forward(self, x1, x2, x3, x4, x5):
		x = self.up1(x5,x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		x = self.up5(x)
		return x

class MultipleDecoder(nn.Module):
	def __inti__(self, in_channels,out_channels):
		super(MultipleDecoder, self).__init__()
		self.enc = 