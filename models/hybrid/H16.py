import torch
import torch.nn as nn
import modules

class H16(nn.Module):
    def __init__(self, d_model, block_layers, input_size, max_len, activation, kernel_sizes, dilations, num_heads, dropout):
        super(H16, self).__init__()
        self.embed = modules.Embedding1D(input_size, d_model, "conv", [4 * d_model], activation)
        self.pe = modules.PositionalEncoding1D(d_model, max_len, convolutional=True)
        
        self.memory = Memory(d_model, activation, num_heads, dropout)
        self.blocks = nn.ModuleList()
        self.memories = nn.ModuleList()
        for kernel_size, dilation in zip(kernel_sizes, dilations):
            padding = modules.compute_padding(kernel_size, 1, dilation)
            self.blocks.append(Block(d_model, block_layers, kernel_size, padding, 1, dilation, activation, dropout))
            self.memories.append(Memory(d_model, activation, num_heads, dropout))
        
        self.unembed = modules.MultiConv1D(d_model, [4 * d_model], 1, 1, 0, 1, 1, activation, dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, logits=False):
        x = self.embed(x)
        mem = self.pe(torch.zeros_like(x))
        x, mem = self.memory(x, mem)
        
        for block, memory in zip(self.blocks, self.memories):
            x = block(x)
            x, mem = memory(x, mem)
        
        x = self.unembed(x)
        x = x.squeeze(1)
        if logits:
            return x
        else:
            x = self.sigmoid(x)
            return x
    
class Memory(nn.Module):
    def __init__(self, d_model, activation, num_heads, dropout):
        super(Memory, self).__init__()
        fn = getattr(nn, activation)
        self.input = nn.Transformer(d_model, num_heads, 1, 1, 4 * d_model, dropout, fn(), batch_first=True)
        self.output = nn.Transformer(d_model, num_heads, 1, 1, 4 * d_model, dropout, fn(), batch_first=True)

    def forward(self, x, mem):
        x = x.permute(0, 2, 1)
        mem = mem.permute(0, 2, 1)
        mem = self.input(x, mem)
        x = self.output(mem, x)
        x = x.permute(0, 2, 1)
        mem = mem.permute(0, 2, 1)
        return x, mem

class Block(nn.Module):
    def __init__(self, d_model, block_layers, kernel_size=3, padding=1, stride=1, dilation=1, activation="GELU", dropout=0.1):
        super(Block, self).__init__()
        layers = list()
        for _ in range(block_layers):
            layers.append(Layer(d_model, kernel_size, padding, stride, dilation, activation, dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

class Layer(nn.Module):
    def __init__(self, d_model, kernel_size=3, padding=1, stride=1, dilation=1, activation="GELU", dropout=0.1):
        super(Layer, self).__init__()
        fn = getattr(nn, activation)
        self.double_conv = modules.MultiConv1D(d_model, [4 * d_model], d_model, kernel_size, padding, stride, dilation, activation, dropout)
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = fn()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        res = self.double_conv(x)
        res = self.dropout(res)
        x = self.norm(res + x)
        x = self.activation(x)
        return x
    
if __name__ == "__main__":
    model = H16(64, 2, 8, 1024, "GELU", [3, 3, 5, 5, 3, 3], [1, 2, 3, 3, 2, 1], 8, 0.1).to(torch.device("cuda"))
    x = torch.randn(16, 8, 1000).to(torch.device("cuda"))
    y = model(x)
    print(y.shape)
