import torch
import torch.nn as nn
import modules

class H16(nn.Module):
    def __init__(self, d_model, input_size, max_len, activation, kernel_sizes, num_heads, dropout):
        super(H16, self).__init__()
        self.embed = modules.Embedding1D(input_size, d_model, "conv", [4 * d_model], activation)
        self.pe = modules.PositionalEncoding1D(d_model, max_len)
        
        self.memory = Memory(d_model, activation, num_heads, dropout)
        self.blocks = nn.ModuleList()
        self.memories = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = modules.compute_padding(kernel_size)
            self.blocks.append(Block(d_model, kernel_size, padding, 1, activation, dropout))
            self.memories.append(Memory(d_model, activation, num_heads, dropout))
        
        self.unembed = modules.MLP(d_model, [4 * d_model], 1, activation)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, logits=False):
        x = self.embed(x)
        mem = self.pe(torch.zeros_like(x).permute(0, 2, 1))
        mem = self.memory(x, mem)
        
        for block, memory in zip(self.blocks, self.memories):
            x = block(x)
            mem = memory(x, mem)
        
        x = self.unembed(mem)
        x = x.squeeze(2)
        if logits:
            return x
        else:
            x = self.sigmoid(x)
            return x
    
class Memory(nn.Module):
    def __init__(self, d_model, activation, num_heads, dropout):
        super(Memory, self).__init__()
        fn = getattr(nn, activation)
        self.attn = nn.TransformerDecoderLayer(d_model, num_heads, 4 * d_model, activation=fn(), batch_first=True, dropout=dropout)

    def forward(self, x, mem):
        x = x.permute(0, 2, 1)
        mem = self.attn(x, mem)
        return mem

class Block(nn.Module):
    def __init__(self, d_model, kernel_size=3, padding=1, stride=1, activation="GELU", dropout=0.1):
        super(Block, self).__init__()
        fn = getattr(nn, activation)
        self.conv1 = modules.MultiConv1D(d_model, [4 * d_model], d_model, kernel_size, padding, stride, activation=activation, dropout=dropout)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.conv2 = modules.MultiConv1D(d_model, [4 * d_model], d_model, kernel_size, padding, stride, activation=activation, dropout=dropout)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.activation = fn()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        res = self.conv1(x)
        x = self.norm1(self.dropout(res) + x)
        x = self.activation(x)
        res = self.conv2(x)
        x = self.norm2(self.dropout(res) + x)
        x = self.activation(x)
        return x
    
if __name__ == "__main__":
    model = H16(96, 7, 4096, "GELU", [3, 5, 5, 3], 4, 0.1).to(torch.device("cuda"))
    x = torch.randn(16, 7, 4000).to(torch.device("cuda"))
    y = model(x)
    print(y.shape)
