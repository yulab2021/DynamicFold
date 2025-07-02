import torch
import torch.nn as nn
import modules
import torch.nn.functional as F

class Main(nn.Module):
    def __init__(self, network_depth, bottleneck_layers, input_size, max_len, d_model, kernel_size, pool_size, channel_rate, num_heads, activation, dropout):
        super(Main, self).__init__()
        self.embed = modules.Embedding1D(input_size, d_model, "MLP", [4 * d_model], activation)
        self.gru = nn.GRU(d_model, d_model, batch_first=True, bidirectional=True)

        in_channels = 2 * d_model
        self.contractions = nn.ModuleList()
        for _ in range(network_depth):
            out_channels = in_channels * channel_rate
            self.contractions.append(ContractionLayer(in_channels, out_channels, activation, kernel_size, pool_size, dropout))
            in_channels = out_channels

        self.bottleneck = BottleneckLayer(in_channels, bottleneck_layers, max_len, activation, num_heads, dropout)

        self.expansions = nn.ModuleList()
        for _ in range(network_depth):
            out_channels = in_channels // channel_rate
            self.expansions.append(ExpansionLayer(in_channels, out_channels, activation, kernel_size, pool_size, dropout))
            in_channels = out_channels

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        x, _ = self.gru(x)

        encs = list()
        for contraction in self.contractions:
            x, enc = contraction(x)
            encs.append(enc)
        x = self.bottleneck(x)
        for expansion, enc in zip(self.expansions, reversed(encs)):
            x = expansion(x, enc)
        return x

class ContractionLayer(nn.Module):
    def __init__(self, input_channels, output_channels, activation, kernel_size, pool_size, dropout):
        super(ContractionLayer, self).__init__()
        fn = getattr(nn, activation)
        padding = modules.compute_padding(kernel_size)
        self.conv = modules.MultiConv1D(input_channels, [output_channels], output_channels, kernel_size, padding, activation=activation, dropout=dropout)
        self.norm = nn.BatchNorm1d(output_channels)
        self.activation = fn()
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        enc = self.activation(self.norm(self.conv(x)))
        x = self.pool(enc)
        x = x.permute(0, 2, 1)
        enc = enc.permute(0, 2, 1)
        return x, enc
    
class BottleneckLayer(nn.Module):
    def __init__(self, num_channels, bottleneck_layers, max_len, activation, num_heads, dropout):
        super(BottleneckLayer, self).__init__()
        fn = getattr(nn, activation)
        self.pe = modules.PositionalEncoding1D(num_channels, max_len)
        layer = nn.TransformerEncoderLayer(num_channels, num_heads, 4 * num_channels, activation=fn(), batch_first=True, dropout=dropout)
        self.attn = nn.TransformerEncoder(layer, bottleneck_layers)

    def forward(self, x):
        x = self.pe(x)
        x = self.attn(x)
        return x
    
class ExpansionLayer(nn.Module):
    def __init__(self, input_channels, output_channels, activation, kernel_size, pool_size, dropout):
        super(ExpansionLayer, self).__init__()
        fn = getattr(nn, activation)
        k, p = modules.compute_convt(pool_size)
        self.unpool = nn.ConvTranspose1d(input_channels, input_channels, kernel_size=k, stride=pool_size, padding=p)
        self.norm1 = nn.BatchNorm1d(input_channels)
        padding = modules.compute_padding(kernel_size)
        self.conv = modules.MultiConv1D(2 * input_channels, [output_channels], output_channels, kernel_size, padding, activation=activation, dropout=dropout)
        self.norm2 = nn.BatchNorm1d(output_channels)
        self.activation = fn()

    def forward(self, x, enc):
        x = x.permute(0, 2, 1)
        enc = enc.permute(0, 2, 1)
        x = self.unpool(x)
        x = F.interpolate(x, size=enc.size(2), mode="linear")
        x = self.activation(self.norm1(x))
        cat = torch.cat([x, enc], dim=1)
        x = self.activation(self.norm2(self.conv(cat)))
        x = x.permute(0, 2, 1)
        return x
    
class MapUnembed(nn.Module):
    def __init__(self, d_model, activation):
        super(MapUnembed, self).__init__()
        self.unembed = modules.MLP(2 * d_model, [4 * d_model], d_model, activation)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.unembed(x)
        x = torch.matmul(x, x.permute(0, 2, 1))
        x = self.sigmoid(x)
        return x
    
class SeqUnembed(nn.Module):
    def __init__(self, d_model, activation):
        super(SeqUnembed, self).__init__()
        self.unembed = modules.MLP(2 * d_model, [4 * d_model], 1, activation)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.unembed(x)
        x = x.squeeze(2)
        x = self.sigmoid(x)
        return x
    
class M01(nn.Module):
    def __init__(self, network_depth, bottleneck_layers, input_size, max_len, d_model, kernel_size, pool_size, channel_rate, num_heads, activation, dropout):
        super(M01, self).__init__()
        self.main = Main(network_depth, bottleneck_layers, input_size, max_len, d_model, kernel_size, pool_size, channel_rate, num_heads, activation, dropout)
        self.map = MapUnembed(d_model, activation)
        self.seq = SeqUnembed(d_model, activation)
        self.switch("seq", True, True)

    def switch(self, route:str, main_grad:bool, unembed_grad:bool):
        self.route = route
        if main_grad:
            for param in self.main.parameters():
                param.requires_grad = True
        else:
            for param in self.main.parameters():
                param.requires_grad = False
        
        if unembed_grad and self.route == "seq":
            for param in self.seq.parameters():
                param.requires_grad = True
            for param in self.map.parameters():
                param.requires_grad = False
        elif unembed_grad and self.route == "map":
            for param in self.seq.parameters():
                param.requires_grad = False
            for param in self.map.parameters():
                param.requires_grad = True
        else:
            for param in self.seq.parameters():
                param.requires_grad = False
            for param in self.map.parameters():
                param.requires_grad = False
        
    def forward(self, inputs):
        if self.route == "map":
            x = self.main(inputs)
            x = self.map(x)
        elif self.route == "seq":
            x = self.main(inputs)
            x = self.seq(x)
        else:
            raise ValueError("invalid route")
        return x

if __name__ == "__main__":
    model = M01(3, 3, 7, 4096, 64, 3, 4, 2, 16, "GELU", 0.1).to(torch.device("cuda"))
    x = torch.randn(1, 7, 4000).to(torch.device("cuda"))
    y = model(x)
    print(y.shape)
