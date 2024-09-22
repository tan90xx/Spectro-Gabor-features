import numpy as np
import time
import torch
import torch.nn as nn
from scipy.signal import cheb2ord, cheby2, convolve, decimate, hilbert, lfilter, spectrogram

def apply_along_axis(function, x, axis: int = 0):
    return torch.stack([function(x_i) for x_i in torch.unbind(x, dim=axis)], dim=axis)
#################################### demon and lofar ↓ ##########################################
import torch
import torch.nn.functional as F

def tpsw(signal, npts=None, n=None, p=None, a=None):
    x = signal.clone().detach().requires_grad_(False) # 513, 1501
    if npts is None:
        npts = x.shape[0]
    if n is None:
        n = int(round(npts*.04/2.0+1))
    if p is None:
        p = int(round(n / 8.0 + 1))
    if a is None:
        a = 2.0
    if p > 0:
        h = torch.cat((torch.ones((n-p+1)), torch.zeros(2 * p-1), torch.ones((n-p+1))))
    else:
        h = torch.ones((1, 2*n+1))
        p = 1
    h /= torch.norm(h, p=1)
    
    def apply_on_spectre(xs):
        # Create convolutional layer with appropriate padding and stride
        conv_layer = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=h.shape[0], padding=h.shape[0]-1, stride=1)
        conv_layer.to('cuda')
        # Set weights of convolutional layer to filter coefficients
        with torch.no_grad():
            conv_layer.weight[:] = h.reshape(1, 1, -1)

        # Apply convolution to input tensor
        #xs = xs.permute(1, 0)  # PyTorch expects dimensions to be [batch_size, input_dim, sequence_length]
        xs = xs.view(1,1,-1)
        out = conv_layer(xs).squeeze()  # Add batch dimension and remove it after convolution

        # Return output tensor with appropriate dimensions
        return out # Undo permutation to match original dimensions
    mx = apply_along_axis(apply_on_spectre, x, axis=1) # 513, 1501
    ix = int(np.floor((h.shape[0] + 1)/2.0))  # 滤波器滞后
    mx = mx[ix-1:npts+ix-1]  # 校正滞后 # 502, 1501
    # 修正光谱的极值点
    ixp = ix - p
    mult = (2 * ixp / torch.cat((torch.ones(p - 1) * ixp, torch.arange(ixp, 2 * ixp + 1)), dim=0)[:, None]).to('cuda')  # 极值点校正 12, 1
    temp = torch.ones((1, x.shape[1])).to('cuda') # 1, 1501
    mx[:ix, :] = mx[:ix, :]*(mult @ temp)  # 起始点
    ###################################################################################
    test = mx[npts-ix:npts, :]
    mx[npts-ix:npts, :] = mx[npts-ix:npts, :]*(torch.flipud(mult) @ temp)  # Pontos finais
    # 消除第二步过滤的峰值
    # indl= torch.where((x-a*mx) > 0) # 点大于a*mx
    indl = (x-a*mx) > 0
    #x[indl] = mx[indl]
    x = torch.where(indl, mx, x)
    mx = apply_along_axis(apply_on_spectre, x, axis=1)
    mx = mx[ix-1:npts+ix-1, :]
    # 修正光谱的极值点
    mx[:ix, :] = mx[:ix, :]*(mult @ temp)  # 起始点
    mx[npts-ix:npts, :] = mx[npts-ix:npts, :]*(torch.flipud(mult) @ temp)  # Pontos finais

    if signal.ndim == 1:
        mx = mx[:, 0]
    return mx.clone().detach().requires_grad_(False)
#################################### demon and lofar ↑ ##########################################

def power_to_db(input):
    r"""Power to db, this function is the pytorch implementation of 
    librosa.power_to_lb
    """
    ref_value = 1.0
    amin = 1e-10
    top_db = 80.0

    log_spec = 10.0 * torch.log10(torch.clamp(input, min=amin, max=np.inf))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise librosa.util.exceptions.ParameterError('top_db must be non-negative')
        log_spec = torch.clamp(log_spec, min=log_spec.max().item() - top_db, max=np.inf)

    return log_spec


def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def do_mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes 
    (1, 3, 5, ...).

    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)

    Returns:
      out: (batch_size, ...)
    """
    out = (x[0 :: 2].transpose(0, -1) * mixup_lambda[0 :: 2] + \
        x[1 :: 2].transpose(0, -1) * mixup_lambda[1 :: 2]).transpose(0, -1)
    return out
    

def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]


def forward(model, generator, return_input=False, 
    return_target=False):
    """Forward data to a model.
    
    Args: 
      model: object
      generator: object
      return_input: bool
      return_target: bool

    Returns:
      audio_name: (audios_num,)
      clipwise_output: (audios_num, classes_num)
      (ifexist) segmentwise_output: (audios_num, segments_num, classes_num)
      (ifexist) framewise_output: (audios_num, frames_num, classes_num)
      (optional) return_input: (audios_num, segment_samples)
      (optional) return_target: (audios_num, classes_num)
    """
    output_dict = {}
    device = next(model.parameters()).device
    time1 = time.time()

    # Forward data to a model in mini-batches
    for n, batch_data_dict in enumerate(generator):
        print(n)
        batch_waveform = move_data_to_device(batch_data_dict['waveform'], device)
        
        with torch.no_grad():
            model.eval()
            batch_output = model(batch_waveform)

        append_to_dict(output_dict, 'audio_name', batch_data_dict['audio_name'])

        append_to_dict(output_dict, 'clipwise_output', 
            batch_output['clipwise_output'].data.cpu().numpy())

        if 'segmentwise_output' in batch_output.keys():
            append_to_dict(output_dict, 'segmentwise_output', 
                batch_output['segmentwise_output'].data.cpu().numpy())

        if 'framewise_output' in batch_output.keys():
            append_to_dict(output_dict, 'framewise_output', 
                batch_output['framewise_output'].data.cpu().numpy())
            
        if return_input:
            append_to_dict(output_dict, 'waveform', batch_data_dict['waveform'])
            
        if return_target:
            if 'target' in batch_data_dict.keys():
                append_to_dict(output_dict, 'target', batch_data_dict['target'])

        if n % 10 == 0:
            print(' --- Inference time: {:.3f} s / 10 iterations ---'.format(
                time.time() - time1))
            time1 = time.time()

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    return output_dict


def interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the 
    resolution reduction in downsampling of a CNN.
    
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate

    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames. The pad value 
    is the same as the value of the last frame.

    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad

    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1 :, :].repeat(1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, audio_length):
    """Count flops. Code modified from others' implementation.
    """
    multiply_adds = True
    list_conv2d=[]
    def conv2d_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
 
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0
 
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
 
        list_conv2d.append(flops)

    list_conv1d=[]
    def conv1d_hook(self, input, output):
        batch_size, input_channels, input_length = input[0].size()
        output_channels, output_length = output[0].size()
 
        kernel_ops = self.kernel_size[0] * (self.in_channels / self.groups) * (2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0
 
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_length
 
        list_conv1d.append(flops)
 
    list_linear=[] 
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
 
        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()
 
        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)
 
    list_bn=[] 
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)
 
    list_relu=[] 
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement() * 2)
 
    list_pooling2d=[]
    def pooling2d_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
 
        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
 
        list_pooling2d.append(flops)

    list_pooling1d=[]
    def pooling1d_hook(self, input, output):
        batch_size, input_channels, input_length = input[0].size()
        output_channels, output_length = output[0].size()
 
        kernel_ops = self.kernel_size[0]
        bias_ops = 0
        
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_length
 
        list_pooling2d.append(flops)
 
    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, nn.Conv2d):
                net.register_forward_hook(conv2d_hook)
            elif isinstance(net, nn.Conv1d):
                net.register_forward_hook(conv1d_hook)
            elif isinstance(net, nn.Linear):
                net.register_forward_hook(linear_hook)
            elif isinstance(net, nn.BatchNorm2d) or isinstance(net, nn.BatchNorm1d):
                net.register_forward_hook(bn_hook)
            elif isinstance(net, nn.ReLU):
                net.register_forward_hook(relu_hook)
            elif isinstance(net, nn.AvgPool2d) or isinstance(net, nn.MaxPool2d):
                net.register_forward_hook(pooling2d_hook)
            elif isinstance(net, nn.AvgPool1d) or isinstance(net, nn.MaxPool1d):
                net.register_forward_hook(pooling1d_hook)
            else:
                print('Warning: flop of module {} is not counted!'.format(net))
            return
        for c in childrens:
            foo(c)

    # Register hook
    foo(model)
    
    device = device = next(model.parameters()).device
    input = torch.rand(1, audio_length).to(device)

    out = model(input)
 
    total_flops = sum(list_conv2d) + sum(list_conv1d) + sum(list_linear) + \
        sum(list_bn) + sum(list_relu) + sum(list_pooling2d) + sum(list_pooling1d)
    
    return total_flops