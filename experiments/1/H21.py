import torch
import torch.nn as nn
import modules
import torch.nn.functional as F

class H21(nn.Module):
    def __init__(self, network_depth, bottleneck_layers, input_size, max_len, seq_channels, map_channels, map_rank, kernel_size, pool_size, channel_rate, num_heads, temperature, activation, dropout):
        super(H21, self).__init__()
        self.embed = modules.Embedding1D(input_size, seq_channels, "MLP", [4 * seq_channels], activation)
        self.gru = nn.GRU(seq_channels, seq_channels, batch_first=True, bidirectional=True)

        in_channels = 2 * seq_channels
        self.contractions = nn.ModuleList()
        for _ in range(network_depth):
            out_channels = in_channels * channel_rate
            self.contractions.append(ContractionLayer(in_channels, out_channels, activation, kernel_size, pool_size, dropout))
            in_channels = out_channels

        out_channels = in_channels * channel_rate
        self.map_ = Map(in_channels, map_channels, max_len, map_rank, activation)
        self.bottleneck = BottleneckLayer(map_channels, bottleneck_layers, activation, num_heads, dropout)
        self.seq = Seq(in_channels, map_channels, out_channels, temperature, map_rank, activation)
        in_channels = out_channels

        self.expansions = nn.ModuleList()
        for _ in range(network_depth):
            out_channels = in_channels // channel_rate
            self.expansions.append(ExpansionLayer(in_channels, out_channels, activation, kernel_size, pool_size, dropout))
            in_channels = out_channels

        self.unembed = modules.MLP(in_channels, [4 * in_channels], 1, activation)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, logits=False):
        x = x.permute(0, 2, 1)
        seq = self.embed(x)
        seq, _ = self.gru(seq)
        seq = seq.permute(0, 2, 1)

        encs = list()
        for contraction in self.contractions:
            seq, enc = contraction(seq)
            encs.append(enc)

        seq = seq.permute(0, 2, 1)
        map_ = self.map_(seq)
        map_ = self.bottleneck(map_)
        seq = self.seq(map_, seq)
        seq = seq.permute(0, 2, 1)

        for expansion, enc in zip(self.expansions, reversed(encs)):
            seq = expansion(seq, enc)

        seq = seq.permute(0, 2, 1)
        x = self.unembed(seq)
        x = x.squeeze(2)
        if logits:
            return x
        else:
            x = self.sigmoid(x)
            return x

class Seq(nn.Module):
    def __init__(self, seq_channels, map_channels, out_channels, temperature, map_rank, activation):
        super(Seq, self).__init__()
        self.denominator = temperature * map_rank**0.5
        self.query = nn.Linear(seq_channels, map_rank)
        self.key = nn.Linear(seq_channels, map_rank)
        self.value = nn.Linear(map_channels, map_channels)
        self.softmax = nn.Softmax(2)
        self.norm = nn.LayerNorm(map_channels)
        self.mlp = modules.MLP(map_channels, [4 * out_channels], out_channels, activation)

    def forward(self, map_, seq):
        query = self.query(seq)
        key = self.key(seq)
        value = self.value(map_.permute(0, 2, 3, 1))
        attn = torch.matmul(query, key.permute(0, 2, 1)).unsqueeze(3)
        attn = self.softmax(attn / self.denominator)
        score = self.norm(torch.sum(value * attn, dim=2))
        score = self.mlp(score)
        return score
    
class Map(nn.Module):
    def __init__(self, seq_channels, map_channels, max_len, map_rank, activation):
        super(Map, self).__init__()
        fn = getattr(nn, activation)
        self.queries = nn.ModuleList()
        self.keys = nn.ModuleList()
        for _ in range(map_channels):
            self.queries.append(nn.Linear(seq_channels, map_rank))
            self.keys.append(nn.Linear(seq_channels, map_rank))
        self.pe = modules.PositionalEncoding2D(map_channels, max_len)
        self.norm = nn.BatchNorm2d(map_channels)
        self.activation = fn()

    def forward(self, seq):
        queries = torch.cat([query(seq).unsqueeze(1) for query in self.queries], dim=1)
        keys = torch.cat([key(seq).unsqueeze(1) for key in self.keys], dim=1)
        map_ = torch.matmul(keys, queries.permute(0, 1, 3, 2))
        map_ = self.activation(self.norm(self.pe(map_)))
        return map_

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
        enc = self.activation(self.norm(self.conv(x)))
        x = self.pool(enc)
        return x, enc
    
class BottleneckLayer(nn.Module):
    def __init__(self, num_channels, bottleneck_layers, activation, num_heads, dropout):
        super(BottleneckLayer, self).__init__()
        fn = getattr(nn, activation)
        layer = nn.TransformerEncoderLayer(num_channels, num_heads, 4 * num_channels, activation=fn(), batch_first=True, dropout=dropout)
        self.attn = nn.TransformerEncoder(layer, bottleneck_layers)

    def forward(self, x):
        _, _, height, width = x.size()
        x = torch.flatten(x.permute(0, 2, 3, 1), 1, 2)
        x = self.attn(x)
        x = torch.unflatten(x, 1, (height, width)).permute(0, 3, 1, 2)
        return x
    
class ExpansionLayer(nn.Module):
    def __init__(self, input_channels, output_channels, activation, kernel_size, pool_size, dropout):
        super(ExpansionLayer, self).__init__()
        fn = getattr(nn, activation)
        k, p = modules.compute_convt(pool_size)
        self.unpool = nn.ConvTranspose1d(input_channels, output_channels, kernel_size=k, stride=pool_size, padding=p)
        self.norm1 = nn.BatchNorm1d(output_channels)
        padding = modules.compute_padding(kernel_size)
        self.conv = modules.MultiConv1D(2 * output_channels, [output_channels], output_channels, kernel_size, padding, activation=activation, dropout=dropout)
        self.norm2 = nn.BatchNorm1d(output_channels)
        self.activation = fn()

    def forward(self, x, enc):
        x = self.unpool(x)
        x = F.interpolate(x, size=enc.size(2), mode="linear")
        x = self.activation(self.norm1(x))
        cat = torch.cat([x, enc], dim=1)
        x = self.activation(self.norm2(self.conv(cat)))
        return x

if __name__ == "__main__":
    model = H21(3, 3, 7, 4096, 64, 64, 64, 3, 4, 2, 16, 1, "GELU", 0.1).to(torch.device("cuda"))
    x = torch.randn(16, 7, 4000).to(torch.device("cuda"))
    y = model(x)
    print(y.shape)
