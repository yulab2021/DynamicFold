import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def compute_padding(kernel_size, stride=1, dilation=1, input_len=None, output_len=None):
    if (input_len is not None) and (output_len is not None):
        total_padding = (output_len - 1) * stride + (dilation * (kernel_size - 1) + 1) - input_len
    else:
        total_padding = (kernel_size - 1) * dilation - stride + 1
    if total_padding % 2 == 0:
        padding = total_padding // 2
    else:
        raise ValueError("padding is not an integer")
    return padding

def compute_conv(stride):
    if stride % 2 == 0:
        kernel_size = 2 * stride
        padding = stride // 2
    else:
        kernel_size = 2 * stride + 1
        padding = (stride + 1) // 2
    return kernel_size, padding

def compute_convt(stride):
    if stride % 2 == 1:
        kernel_size = 2 * stride - 1
        padding = (stride - 1) // 2
    else:
        kernel_size = 2 * stride
        padding = stride // 2
    return kernel_size, padding

class Permutation(nn.Module):
    def __init__(self, dimensions):
        super(Permutation, self).__init__()
        self.dimensions = dimensions
    
    def forward(self, x):
        return torch.permute(x, self.dimensions)

class Folder1D:
    def __init__(self, kernel_size, padding=0, stride=1, dilation=1):
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.window_total = (self.kernel_size - 1) * self.dilation + 1

    def unfold(self, x):
        if x.dim() == 2:
            length, num_channels = x.size()
            batch_size = 1
            num_dim = 2
            x = x.unsqueeze(0)
        elif x.dim() == 3:
            batch_size, length, num_channels = x.size()
            num_dim = 3
        else:
            raise ValueError("input must be 2D or 3D tensor")
        if length <= self.window_total:
            flattened = True
        else:
            x = x.permute(0, 2, 1).unsqueeze(3)
            x = F.unfold(x, (self.kernel_size, 1), (self.dilation, 1), (self.padding, 0), (self.stride, 1)).permute(0, 2, 1)
            x = x.reshape(-1, num_channels, self.kernel_size).permute(0, 2, 1)
            flattened = False
        total = x.numel()
        sizes = (batch_size, num_channels, length, num_dim, total, flattened)
        return x, sizes
    
    def fold(self, x, sizes):
        batch_size, num_channels, length, num_dim, total, flattened = sizes
        if flattened:
            if (x.dim() != 3) or (x.shape[1:] != (length, num_channels)) or (x.numel() != total):
                raise ValueError("input tensor does not match expected shape")
        else:
            if (x.dim() != 3) or (x.shape[1:] != (self.kernel_size, num_channels)) or (x.numel() != total):
                raise ValueError("input tensor does not match expected shape")
            x = x.permute(0, 2, 1).reshape(batch_size, -1, num_channels * self.kernel_size)
            x = F.fold(x.permute(0, 2, 1), (length, 1), (self.kernel_size, 1), (self.dilation, 1), (self.padding, 0), (self.stride, 1))
            x = x.squeeze(3).permute(0, 2, 1)
        if num_dim == 2:
            x = x.squeeze(0)
        return x
    
class Folder2D:
    def __init__(self, kernel_size, padding=0, stride=1, dilation=1):
        self.kernel_size = self.convert_arg(kernel_size)
        self.padding = self.convert_arg(padding)
        self.stride = self.convert_arg(stride)
        self.dilation = self.convert_arg(dilation)
        self.kernel_total = self.kernel_size[0] * self.kernel_size[1]
        self.window_total = ((self.kernel_size[0] - 1) * self.dilation[0] + 1) * ((self.kernel_size[1] - 1) * self.dilation[1] + 1)
        
    def convert_arg(self, arg):
        if type(arg) is int:
            return (arg, arg)
        elif (type(arg) is tuple) and (len(arg) == 2):
            return arg
        else:
            raise ValueError("invalid argument type")

    def unfold(self, x):
        if x.dim() == 3:
            num_channels, height, width = x.size()
            batch_size = 1
            num_dim = 3
            x = x.unsqueeze(0)
        if x.dim() == 4:
            batch_size, num_channels, height, width = x.size()
            num_dim = 4
        else:
            raise ValueError("input must be 3D or 4D tensor")
        if height * width <= self.window_total:
            x = x.permute(0, 2, 3, 1)
            x = torch.flatten(x, 1, 2)
            flattened = True
        else:
            x = F.unfold(x, self.kernel_size, self.dilation, self.padding, self.stride).permute(0, 2, 1)
            x = x.reshape(-1, num_channels, self.kernel_total).permute(0, 2, 1)
            flattened = False
        total = x.numel()
        sizes = (batch_size, num_channels, height, width, num_dim, total, flattened)
        return x, sizes
    
    def fold(self, x, sizes):
        batch_size, num_channels, height, width, num_dim, total, flattened = sizes
        if flattened:
            if (x.dim() != 3) or (x.shape[1:] != (height * width, num_channels)) or (x.numel() != total):
                raise ValueError("input tensor does not match expected shape")
            x = torch.unflatten(x, 1, (height, width))
            x = x.permute(0, 3, 1, 2)
        else:
            if (x.dim() != 3) or (x.shape[1:] != (self.kernel_total, num_channels)) or (x.numel() != total):
                raise ValueError("input tensor does not match expected shape")
            x = x.permute(0, 2, 1).reshape(batch_size, -1, num_channels * self.kernel_total)
            x = F.fold(x.permute(0, 2, 1), (height, width), self.kernel_size, self.dilation, self.padding, self.stride)
        if num_dim == 3:
            x = x.squeeze(0)
        return x
    
class BatchWrapper(nn.Module):
    def __init__(self, layer, batch_size):
        super(BatchWrapper, self).__init__()
        self.layer = layer
        self.batch_size = batch_size

    def forward(self, x):
        outputs = list()
        if isinstance(x, (list, tuple)):
            num_param = len(x)
            num_batches = (x[0].size(0) - 1) // self.batch_size + 1
            for i in range(num_batches):
                param = [x[j][(i * self.batch_size):((i + 1) * self.batch_size)] for j in range(num_param)]
                outputs.append(self.layer(*param))
        else:
            num_batches = (x.size(0) - 1) // self.batch_size + 1
            for i in range(num_batches):
                outputs.append(self.layer(x[(i * self.batch_size):((i + 1) * self.batch_size)]))
        
        return torch.cat(outputs, dim=0)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation):
        super(MLP, self).__init__()
        fn = getattr(nn, activation)
        layers = []
        in_size = input_size
        for out_size in hidden_size:
            layers.append(nn.Linear(in_size, out_size))
            layers.append(fn())
            in_size = out_size
        layers.append(nn.Linear(in_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class MultiConv1D(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, kernel_size=3, padding=1, stride=1, dilation=1, activation="ReLU", dropout=0.1):
        super(MultiConv1D, self).__init__()
        fn = getattr(nn, activation)
        layers = []
        in_channels = input_channels
        for out_channels in hidden_channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                fn(),
                nn.Dropout(p=dropout)
            ])
            in_channels = out_channels
        layers.append(nn.Conv1d(in_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class MultiConv2D(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, kernel_size=3, padding=1, stride=1, dilation=1, activation="ReLU", dropout=0.1):
        super(MultiConv2D, self).__init__()
        fn = getattr(nn, activation)
        layers = []
        in_channels = input_channels
        for out_channels in hidden_channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
                nn.BatchNorm2d(out_channels),
                fn(),
                nn.Dropout(p=dropout)
            ])
            in_channels = out_channels
        layers.append(nn.Conv2d(in_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len, convolutional=False):
        super(PositionalEncoding1D, self).__init__()
        self.convolutional = convolutional
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.convolutional:
            x = x + self.pe[:, :x.size(2), :].permute(0, 2, 1)
        else:
            x = x + self.pe[:, :x.size(1), :]
        return x

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding2D, self).__init__()
        pe = torch.zeros(max_len, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 4) * (-np.log(10000.0) / d_model))
        pe[:, :, 0::4] = torch.sin(position * div_term).unsqueeze(0)
        pe[:, :, 1::4] = torch.cos(position * div_term).unsqueeze(0)
        pe[:, :, 2::4] = torch.sin(position * div_term).unsqueeze(1)
        pe[:, :, 3::4] = torch.cos(position * div_term).unsqueeze(1)
        pe = pe.unsqueeze(0).permute(0, 3, 1, 2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :, :x.size(2), :x.size(3)]
        return x

class Embedding1D(nn.Module):
    def __init__(self, input_size, output_size, embedding="linear", hidden_size=None, activation=None, encoding=None, max_len=None):
        super(Embedding1D, self).__init__()
        if embedding == "MLP":
            embedding = MLP(input_size, hidden_size, output_size, activation)
            norm = nn.LayerNorm(output_size)
        elif embedding == "conv":
            embedding = MultiConv1D(input_size, hidden_size, output_size, 1, 0, 1, 1, activation)
            norm = nn.BatchNorm1d(output_size)
        elif embedding == "linear":
            embedding = nn.Linear(input_size, output_size)
            norm = nn.Identity()
        else:
            raise ValueError("invalid embedding method")

        if encoding == "pos":
            encoding = PositionalEncoding1D(output_size, max_len, embedding=="conv")
        else:
            encoding = nn.Identity()
        
        self.model = nn.Sequential(
            embedding,
            encoding,
            norm
        )
    
    def forward(self, x):
        return self.model(x)
