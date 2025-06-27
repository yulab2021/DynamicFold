import torch
import torch.nn as nn
import modules
import torch.nn.functional as F

class H19(nn.Module):
    def __init__(self, network_depth, bottleneck_layers, input_size, max_len, d_model, patch_size, pool_size, channel_rate, num_heads, batch_size, activation, dropout):
        super(H19, self).__init__()
        self.embed = modules.Embedding1D(input_size, d_model, "MLP", [4 * d_model], activation, encoding="pos", max_len=max_len)
        self.gru = nn.GRU(d_model, d_model, batch_first=True, bidirectional=True)

        in_channels = 2 * d_model
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

        self.unembed = modules.MLP(2 * d_model, [4 * d_model], 1, activation)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, logits=False):
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

        x = self.unembed(x)
        x = x.squeeze(2)
        if logits:
            return x
        else:
            x = self.sigmoid(x)
            return x

class ContractionLayer(nn.Module):
    def __init__(self, input_channels, output_channels, activation, patch_size, pool_size, num_heads, batch_size, dropout):
        super(ContractionLayer, self).__init__()
        fn = getattr(nn, activation)
        padding = modules.compute_padding(patch_size, patch_size)
        self.folder = modules.Folder1D(patch_size, padding, patch_size)
        self.attn = modules.BatchWrapper(nn.TransformerEncoderLayer(input_channels, num_heads, 4 * input_channels, activation=fn(), batch_first=True, dropout=dropout), batch_size)
        padding = modules.compute_padding(pool_size, pool_size)
        self.pool = nn.Conv1d(input_channels, output_channels, pool_size, pool_size, padding)

    def forward(self, x):
        x, sizes = self.folder.unfold(x)
        enc = self.attn(x)
        enc = self.folder.fold(enc, sizes)
        x = self.pool(enc.permute(0, 2, 1)).permute(0, 2, 1)
        return x, enc
    
class BottleneckLayer(nn.Module):
    def __init__(self, num_channels, bottleneck_layers, activation, num_heads, batch_size, dropout):
        super(BottleneckLayer, self).__init__()
        fn = getattr(nn, activation)
        layer = nn.TransformerEncoderLayer(num_channels, num_heads, 4 * num_channels, activation=fn(), batch_first=True, dropout=dropout)
        self.attn = modules.BatchWrapper(nn.TransformerEncoder(layer, bottleneck_layers), batch_size)

    def forward(self, x):
        x = self.attn(x)
        return x
    
class ExpansionLayer(nn.Module):
    def __init__(self, input_channels, output_channels, activation, patch_size, pool_size, num_heads, batch_size, dropout):
        super(ExpansionLayer, self).__init__()
        fn = getattr(nn, activation)
        kernel_size, padding = modules.compute_convt(pool_size)
        self.unpool = nn.ConvTranspose1d(input_channels, output_channels, kernel_size=kernel_size, stride=pool_size, padding=padding)
        padding = modules.compute_padding(patch_size, patch_size)
        self.folder = modules.Folder1D(patch_size, padding, patch_size)
        self.attn = modules.BatchWrapper(nn.TransformerDecoderLayer(output_channels, num_heads, 4 * output_channels, activation=fn(), batch_first=True, dropout=dropout), batch_size)

    def forward(self, x, enc):
        x = self.unpool(x.permute(0, 2, 1))
        x = F.interpolate(x, size=enc.size(1), mode="linear").permute(0, 2, 1)
        x, sizes = self.folder.unfold(x)
        enc, _ = self.folder.unfold(enc)
        x = self.attn([x, enc])
        x = self.folder.fold(x, sizes)
        return x

if __name__ == "__main__":
    model = H19(3, 3, 7, 4096, 96, 16, 4, 2, 4, 65536, "GELU", 0.1).to(torch.device("cuda"))
    x = torch.randn(8, 7, 4000).to(torch.device("cuda"))
    y = model(x)
    print(y.shape)
