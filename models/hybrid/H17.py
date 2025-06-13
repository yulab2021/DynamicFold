import torch
import torch.nn as nn
import modules
import torch.nn.functional as F

class H17(nn.Module):
    def __init__(self, num_blocks, block_depth, num_layers, input_size, max_len, seq_channels, map_channels, map_rank, patch_size, pool_size, channel_rate, compression_rate, activation, num_heads, temperature, dropout):
        super(H17, self).__init__()
        self.embed = modules.Embedding1D(input_size, seq_channels, "MLP", [4 * seq_channels], activation)
        blocks = list()
        for _ in range(num_blocks):
            blocks.append(Block(seq_channels, map_channels, max_len, map_rank, block_depth, num_layers, patch_size, pool_size, channel_rate, compression_rate, activation, num_heads, temperature, dropout))
        self.blocks = nn.Sequential(*blocks)
        self.unembed = modules.MLP(seq_channels, [4 * seq_channels], 1, activation)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, logits=False):
        x = x.permute(0, 2, 1)
        seq = self.embed(x)
        seq = self.blocks(seq)
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
        kernel_size, padding = modules.compute_convt(compression_rate)
        self.attn = nn.ConvTranspose2d(source_channels, source_channels, kernel_size, compression_rate, padding)
        self.softmax = nn.Softmax(2)
        self.values = nn.ModuleList()
        for _ in range(source_channels):
            self.values.append(nn.Linear(target_channels, target_channels))
        self.sum = nn.Linear(source_channels, 1)
        self.norm = nn.LayerNorm(target_channels)

    def forward(self, source, target):
        attn = self.attn(source)
        attn = self.softmax(attn / self.temperature)
        values = torch.cat([value(target).unsqueeze(1) for value in self.values], dim=1)
        score = torch.matmul(attn, values)
        res = self.sum(score.permute(0, 2, 3, 1)).squeeze(3)
        target = self.norm(res + target)
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
        self.norm = modules.ConvLayerNorm2D(target_channels)
        self.pe = modules.PositionalEncoding2D(target_channels, max_len)

    def forward(self, source):
        source = source.permute(0, 2, 1)
        queries = torch.cat([query(source).unsqueeze(1) for query in self.queries], dim=1)
        keys = torch.cat([key(source).unsqueeze(1) for key in self.keys], dim=1)
        target = torch.matmul(keys.permute(0, 1, 3, 2), queries)
        target = self.norm(target)
        target = self.pe(target)
        return target

class Block(nn.Module):
    def __init__(self, seq_channels, map_channels, max_len, map_rank, block_depth, num_layers, patch_size, pool_size, channel_rate, compression_rate, activation, num_heads, temperature, dropout):
        super(Block, self).__init__()
        self.gru = nn.GRU(seq_channels, seq_channels, batch_first=True, bidirectional=True)
        self.input = MapGate(2 * seq_channels, map_channels, max_len, compression_rate, map_rank)

        in_channels = map_channels
        self.contractions = nn.ModuleList()
        for _ in range(block_depth):
            out_channels = in_channels * channel_rate
            self.contractions.append(ContractionLayer(in_channels, out_channels, activation, patch_size, pool_size, num_heads, dropout))
            in_channels = out_channels

        self.bottleneck = BottleneckLayer(in_channels, num_layers, activation, num_heads, dropout)

        self.expansions = nn.ModuleList()
        for _ in range(block_depth):
            out_channels = in_channels // channel_rate
            self.expansions.append(ExpansionLayer(in_channels, out_channels, activation, patch_size, pool_size, num_heads, dropout))
            in_channels = out_channels

        self.output = SeqGate(map_channels, seq_channels, temperature, compression_rate)
    
    def forward(self, seq):
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
        return seq

class ContractionLayer(nn.Module):
    def __init__(self, input_channels, output_channels, activation, patch_size, pool_size, num_heads, dropout):
        super(ContractionLayer, self).__init__()
        fn = getattr(nn, activation)
        padding = modules.compute_padding(patch_size, patch_size)
        self.folder = modules.Folder2D(patch_size, padding, patch_size)
        self.attn = nn.TransformerEncoderLayer(input_channels, num_heads, 4 * input_channels, activation=fn(), batch_first=True, dropout=dropout)
        padding = modules.compute_padding(pool_size, pool_size)
        self.pool = nn.Conv2d(input_channels, output_channels, pool_size, pool_size, padding)

    def forward(self, x):
        x, sizes = self.folder.unfold(x)
        enc = self.attn(x)
        enc = self.folder.fold(enc, sizes)
        x = self.pool(enc)
        return x, enc
    
class BottleneckLayer(nn.Module):
    def __init__(self, num_channels, num_layers, activation, num_heads, dropout):
        super(BottleneckLayer, self).__init__()
        fn = getattr(nn, activation)
        encoder_layer = nn.TransformerEncoderLayer(num_channels, num_heads, 4 * num_channels, activation=fn(), batch_first=True, dropout=dropout)
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        _, _, height, width = x.size()
        x = torch.flatten(x.permute(0, 2, 3, 1), 1, 2)
        x = self.attn(x)
        x = torch.unflatten(x, 1, (height, width)).permute(0, 3, 1, 2)
        return x
    
class ExpansionLayer(nn.Module):
    def __init__(self, input_channels, output_channels, activation, patch_size, pool_size, num_heads, dropout):
        super(ExpansionLayer, self).__init__()
        fn = getattr(nn, activation)
        kernel_size, padding = modules.compute_convt(pool_size)
        self.unpool = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=pool_size, padding=padding)
        padding = modules.compute_padding(patch_size, patch_size)
        self.folder = modules.Folder2D(patch_size, padding, patch_size)
        self.attn = nn.TransformerDecoderLayer(output_channels, num_heads, 4 * output_channels, activation=fn(), batch_first=True, dropout=dropout)

    def forward(self, x, enc):
        x = self.unpool(x)
        x = F.interpolate(x, size=[enc.size(2), enc.size(3)], mode="bicubic")
        x, sizes = self.folder.unfold(x)
        enc, _ = self.folder.unfold(enc)
        x = self.attn(x, enc)
        x = self.folder.fold(x, sizes)
        return x

if __name__ == "__main__":
    model = H17(1, 4, 3, 8, 1024, 256, 8, 64, 16, 2, 2, 1, "GELU", 4, 2, 0.1).to(torch.device("cuda"))
    x = torch.randn(4, 8, 1000).to(torch.device("cuda"))
    y = model(x)
    print(y.shape)
