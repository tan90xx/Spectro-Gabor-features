"""
Containing functions cloned from https://github.com/qshobak/Transformation-invariant-Gabor-Convolutional-Networks
To make sure not become broken when updating TIGCN
"""
from __future__ import division
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.nn.modules.conv import _ConvNd
from torch.autograd.function import once_differentiable
from torch.autograd import Variable
import numpy as np
import random

class GaborYu(_ConvNd):
	"""My Gabor Convolution Layer"""
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
					padding=1, dilation=1, groups=1, bias=False, expand=False, padding_mode='zeros'):
		if groups != 1:
			raise ValueError('Group-conv not supported!')
		kw = kernel_size
		kh = kernel_size
		kernel_size = _pair(kernel_size)
		stride = _pair(stride)
		padding = _pair(padding)
		dilation = _pair(dilation)
		super(GaborYu, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

		self.kernel_size = kernel_size
		self.weight = nn.Parameter(getGaborFilterBank(*self.kernel_size))
		self.weight.requires_grad = True

	def forward(self, x):	#128,3,32,32
		xGabor = torch.unsqueeze(x, 2) 	#128,3,1,32,32
		xGabor = xGabor.view(-1, 1, x.size(2), x.size(3))		#128*3,1,32,32

		xConv = F.conv2d(xGabor, self.weight, None, self.stride, self.padding, self.dilation, self.groups) 	#128*3,4,32,32
		y = xConv.view(-1, x.size(1), xConv.size(2), xConv.size(3))	#128*4,3,32,32
		return y

class GaborConv(_ConvNd):
	"""My Gabor Convolution Layer"""
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
					padding=0, dilation=1, groups=1, bias=False, expand=False, padding_mode='zeros'):
		if groups != 1:
			raise ValueError('Group-conv not supported!')
		kw = kernel_size
		kh = kernel_size
		kernel_size = _pair(kernel_size)
		stride = _pair(stride)
		padding = _pair(padding)
		dilation = _pair(dilation)
		super(GaborConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

		self.kernel_size = kernel_size

	def forward(self, x):
		# x--128,80,32,32
		with torch.no_grad():
			new_weight = getGaborFilterBank(*self.kernel_size)	#4,1,3,3
		if x.is_cuda:
			new_weight = new_weight.cuda()

		xGabor = torch.unsqueeze(x, 2) 	#128,80,1,32,32
		xGabor = xGabor.view(-1, 1, x.size(2), x.size(3))		#128*80,1,32,32

		xConv = F.conv2d(xGabor, new_weight, None, 1, 1, 1, 1)	#3*3	#128*80,4,32,32
		#xConv = F.conv2d(xGabor, new_weight, None, 1, 2, 1, 1)	#5*5	#128*80,4,32,32

		xConv = xConv.view(-1, x.size(1), x.size(2), x.size(3))	#128*4,80,32,32
		xConv = F.conv2d(xConv, self.weight, None, self.stride, self.padding, self.dilation, self.groups) 	#128*4,160,32,32
		y = xConv.view(x.size(0), -1, xConv.size(1), xConv.size(2), xConv.size(3))	#128,4,160,32,32
		y = torch.max(y, 1)[0]
		return y

class GaborConvSca(_ConvNd):
	"""My Gabor Convolution Layer"""
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
					padding=0, dilation=1, groups=1, bias=False, expand=False, padding_mode='zeros'):
		if groups != 1:
			raise ValueError('Group-conv not supported!')
		kw = kernel_size
		kh = kernel_size
		kernel_size = _pair(kernel_size)
		stride = _pair(stride)
		padding = _pair(padding)
		dilation = _pair(dilation)
		super(GaborConvSca, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

		self.kernel_size = kernel_size

	def forward(self, x):
		with torch.no_grad():
			new_weight = getScaleGaborFilterBank(*self.kernel_size)
		if x.is_cuda:
			new_weight = new_weight.cuda()
		#	self.weight = self.weight.cuda()

		self.extendWeight = torch.cat((self.weight,self.weight,self.weight), 1)

		x = F.conv2d(x, new_weight, None, 1, 3, 1, 1)	#7*7
		y = F.conv2d(x, self.extendWeight, None, self.stride, self.padding, self.dilation, self.groups) / 3
		return y

def getGaborFilterBank(h, w):
	nScale = 1
	M = 4
	Kmax = math.pi / 2
	f = math.sqrt(2)
	k = Kmax / f ** (nScale - 1)
	sigma = math.pi
	sqsigma = sigma ** 2
	postmean = math.exp(-sqsigma / 2)
	if h != 1:
		gfilter_real = torch.zeros(M, h, w)
		for i in range(M):
			theta = i / M * math.pi
			kcos = k * math.cos(theta)
			ksin = k * math.sin(theta)
			for y in range(h):
				for x in range(w):
					y1 = y + 1 - ((h + 1) / 2)
					x1 = x + 1 - ((w + 1) / 2)
					tmp1 = math.exp(-(k * k * (x1 * x1 + y1 * y1) / (2 * sqsigma)))
					tmp2 = math.cos(kcos * x1 + ksin * y1) - postmean # For real part
					gfilter_real[i][y][x] = k * k * tmp1 * tmp2 / sqsigma	
			xymax = torch.max(gfilter_real[i])
			xymin = torch.min(gfilter_real[i])
			gfilter_real[i] = (gfilter_real[i] - xymin) / (xymax - xymin)
	else:
		gfilter_real = torch.ones(M, h, w)
	Gfilter_real = torch.zeros(M, 1, h, w)
	Gfilter_real = torch.unsqueeze(gfilter_real, 1) 
	return Gfilter_real

def getScaleGaborFilterBank(h, w):
	#nScale = 1
	theta = 0
	M = 3	#scale num
	Kmax = math.pi / 2
	f = math.sqrt(2)
	sigma = math.pi
	sqsigma = sigma ** 2
	postmean = math.exp(-sqsigma / 2)
	if h != 1:
		gfilter_real = torch.zeros(M, h, w)
		for i in range(M):
			nScale = i + 1
			k = Kmax / f ** (nScale - 1)
			kcos = k * math.cos(theta)
			ksin = k * math.sin(theta)
			for y in range(h):
				for x in range(w):
					y1 = y + 1 - ((h + 1) / 2)
					x1 = x + 1 - ((w + 1) / 2)
					tmp1 = math.exp(-(k * k * (x1 * x1 + y1 * y1) / (2 * sqsigma)))
					tmp2 = math.cos(kcos * x1 + ksin * y1) - postmean # For real part
					gfilter_real[i][y][x] = k * k * tmp1 * tmp2 / sqsigma	
			xymax = torch.max(gfilter_real[i])
			xymin = torch.min(gfilter_real[i])
			gfilter_real[i] = (gfilter_real[i] - xymin) / (xymax - xymin)
	else:
		gfilter_real = torch.ones(M, h, w)
	Gfilter_real = torch.zeros(M, 1, h, w)
	Gfilter_real = torch.unsqueeze(gfilter_real, 1) 
	return Gfilter_real

def main():
	mconv = GaborYu(1, 8, 3, padding=1, stride=1, bias=False, groups=1, expand=False)
	mconv = mconv.cuda()
	#print('Parameters:',list(mconv.parameters()))
	print('Parameters Size:',mconv.weight.size())
	print('Weight grad:',mconv.weight.grad)
	
	raw_input = torch.ones(128, 1, 32, 32).cuda()
	#raw_input = torch.ones(2, 1, 32, 32)
	#raw_input = torch.ones((2, 1, 32, 32), requires_grad = True).cuda()
	y = mconv(raw_input)
	#print(y.is_cuda)
	print('Output Size:', y.size())
	
	#z = torch.mean(y)
	#z.backward()
	#print('Weight grad after BP:', mconv.weight.grad.size())

if __name__ == '__main__':
	main()
