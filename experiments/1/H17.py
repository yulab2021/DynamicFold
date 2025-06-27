import torch
import torch.nn as nn
import modules
import torch.nn.functional as F

class H17(nn.Module):
    def __init__(self, network_depth, bottleneck_layers, input_size, max_len, seq_channels, map_channels, map_rank, patch_size, pool_size, channel_rate, compression_rate, num_heads, batch_size, temperature, activation, dropout):
        super(H17, self).__init__()
        self.embed = modules.Embedding1D(input_size, seq_channels, "MLP", [4 * seq_channels], activation)

        self.gru = nn.GRU(seq_channels, seq_channels, batch_first=True, bidirectional=True)
        self.input = MapGate(2 * seq_channels, map_channels, max_len, compression_rate, map_rank)

        in_channels = map_channels
        self.contractions = nn.ModuleList()
        for _ in range(network_depth):
            out_channels = in_channels * channel_rate
            self.contractions.append(ContractionLayer(in_channels, out_channels, activation, patch_size, pool_size, num_heads, batch_size, dropout))
            in_channels = out_channels

        self.bottleneck = BottleneckLayer(in_channels, bottleneck_layers, activation, num_heads, batch_size, dropout)

        self.expansions = nn.ModuleList()
        for _ in range(network_depth):
            out_channels = in_channels // channel_rate
            self.expansions.append(ExpansionLayer(in_channels, out_channels, activation, patch_size, pool_size, num_heads, batch_size, dropout))
            in_channels = out_channels

        self.output = SeqGate(map_channels, 2 * seq_channels, temperature, compression_rate)
        self.unembed = modules.MLP(4 * seq_channels, [4 * seq_channels], 1, activation)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, logits=False):
        x = x.permute(0, 2, 1)
        seq = self.embed(x)
        seq, _ = self.gru(seq)
        map_ = self.input(seq)

        encs = list()
        for contraction in self.contractions:
            map_, enc = contraction(map_)
            encs.append(enc)
        map_ = self.bottleneck(map_)
        for expansion, enc in zip(self.expansions, reversed(encs)):
            map_ = expansion(map_, enc)

        seq = self.output(map_, seq)
        x = self.unembed(seq)
        x = x.squeeze(2)
        if logits:
            return x
        else:
            x = self.sigmoid(x)
            return x

class SeqGate(nn.Module):
    def __init__(self, source_channels, target_channels, temperature, compression_rate):
        super(SeqGate, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(2)
        self.values = nn.ModuleList()
        for _ in range(source_channels):
            kernel_size, padding = modules.compute_conv(compression_rate)
            self.values.append(nn.Conv1d(target_channels, target_channels, kernel_size, compression_rate, padding))
        kernel_size, padding = modules.compute_convt(compression_rate)
        self.sum = nn.ConvTranspose1d(source_channels * target_channels, target_channels, kernel_size, compression_rate, padding)
        self.gru = nn.GRU(2 * target_channels, target_channels, batch_first=True, bidirectional=True)

    def forward(self, source, target):
        attn = self.softmax(source / (self.temperature * source.size(1)**0.5))
        values = torch.cat([value(target.permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(1) for value in self.values], dim=1)
        scores = torch.matmul(attn, values)

        scores = self.sum(scores.permute(0, 3, 1, 2).flatten(1, 2))
        scores = F.interpolate(scores, size=target.size(1), mode="linear").permute(0, 2, 1)
        cat = torch.cat([target, scores], dim=2)
        target, _ = self.gru(cat)
        return target
    
class MapGate(nn.Module):
    def __init__(self, source_channels, target_channels, max_len, compression_rate, map_rank):
        super(MapGate, self).__init__()
        self.queries = nn.ModuleList()
        self.keys = nn.ModuleList()
        for _ in range(target_channels):
            kernel_size, padding = modules.compute_conv(compression_rate)
            self.queries.append(nn.Conv1d(source_channels, map_rank, kernel_size, compression_rate, padding))
            self.keys.append(nn.Conv1d(source_channels, map_rank, kernel_size, compression_rate, padding))
        self.norm = nn.BatchNorm2d(target_channels)
        self.pe = modules.PositionalEncoding2D(target_channels, max_len)

    def forward(self, source):
        source = source.permute(0, 2, 1)
        queries = torch.cat([query(source).unsqueeze(1) for query in self.queries], dim=1)
        keys = torch.cat([key(source).unsqueeze(1) for key in self.keys], dim=1)
        target = torch.matmul(keys.permute(0, 1, 3, 2), queries)
        target = self.norm(target)
        target = self.pe(target)
        return target

class ContractionLayer(nn.Module):
    def __init__(self, input_channels, output_channels, activation, patch_size, pool_size, num_heads, batch_size, dropout):
        super(ContractionLayer, self).__init__()
        fn = getattr(nn, activation)
        padding = modules.compute_padding(patch_size, patch_size)
        self.folder = modules.Folder2D(patch_size, padding, patch_size)
        self.attn = modules.BatchWrapper(nn.TransformerEncoderLayer(input_channels, num_heads, 4 * input_channels, activation=fn(), batch_first=True, dropout=dropout), batch_size)
        padding = modules.compute_padding(pool_size, pool_size)
        self.pool = nn.Conv2d(input_channels, output_channels, pool_size, pool_size, padding)

    def forward(self, x):
        x, sizes = self.folder.unfold(x)
        enc = self.attn(x)
        enc = self.folder.fold(enc, sizes)
        x = self.pool(enc)
        return x, enc
    
class BottleneckLayer(nn.Module):
    def __init__(self, num_channels, bottleneck_layers, activation, num_heads, batch_size, dropout):
        super(BottleneckLayer, self).__init__()
        fn = getattr(nn, activation)
        layer = nn.TransformerEncoderLayer(num_channels, num_heads, 4 * num_channels, activation=fn(), batch_first=True, dropout=dropout)
        self.attn = modules.BatchWrapper(nn.TransformerEncoder(layer, bottleneck_layers), batch_size)

    def forward(self, x):
        _, _, height, width = x.size()
        x = torch.flatten(x.permute(0, 2, 3, 1), 1, 2)
        x = self.attn(x)
        x = torch.unflatten(x, 1, (height, width)).permute(0, 3, 1, 2)
        return x
    
class ExpansionLayer(nn.Module):
    def __init__(self, input_channels, output_channels, activation, patch_size, pool_size, num_heads, batch_size, dropout):
        super(ExpansionLayer, self).__init__()
        fn = getattr(nn, activation)
        kernel_size, padding = modules.compute_convt(pool_size)
        self.unpool = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=pool_size, padding=padding)
        self.norm = nn.BatchNorm2d(output_channels)
        padding = modules.compute_padding(patch_size, patch_size)
        self.folder = modules.Folder2D(patch_size, padding, patch_size)
        self.attn = modules.BatchWrapper(nn.TransformerDecoderLayer(output_channels, num_heads, 4 * output_channels, activation=fn(), batch_first=True, dropout=dropout), batch_size)
        self.activation = fn()

    def forward(self, x, enc):
        x = self.unpool(x)
        x = F.interpolate(x, size=[enc.size(2), enc.size(3)], mode="bicubic")
        x = self.activation(self.norm(x))
        x, sizes = self.folder.unfold(x)
        enc, _ = self.folder.unfold(enc)
        x = self.attn([x, enc])
        x = self.folder.fold(x, sizes)
        return x

if __name__ == "__main__":
    model = H17(4, 4, 7, 4096, 256, 16, 64, 16, 2, 2, 4, 4, 65536, 1, "GELU", 0.1).to(torch.device("cuda"))
    x = torch.randn(8, 7, 4000).to(torch.device("cuda"))
    y = model(x)
    print(y.shape)
