import torch
import torch.nn as nn
import torch.nn.functional as F
import modules

class H15(nn.Module):
    def __init__(self, d_model, num_layers, input_size, activation, channel_rate, pool_size, num_heads, dropout):
        super(H15, self).__init__()
        self.embed = modules.Embedding1D(input_size, d_model, "MLP", [4 * d_model], activation)
        self.gru = nn.GRU(d_model, d_model, batch_first=True, bidirectional=True)
        
        self.contractions = nn.ModuleList()
        in_size = 2 * d_model
        for _ in range(num_layers):
            out_size = in_size * channel_rate
            self.contractions.append(ContractionLayer(in_size, out_size, activation, num_heads, "MaxPool1d", pool_size, dropout))
            in_size = out_size

        out_size = in_size * channel_rate
        self.bottleneck = EncoderLayer(in_size, out_size, activation, num_heads, dropout)
        in_size = out_size

        self.expansions = nn.ModuleList()
        for _ in range(num_layers):
            out_size = in_size // channel_rate
            self.expansions.append(ExpansionLayer(in_size, out_size, activation, pool_size, num_heads, dropout))
            in_size = out_size
        
        self.unembed = modules.MultiConv1D(in_size, [4 * in_size], 1, 1, 0, 1, 1, activation, dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, logits=False):
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        x, _ = self.gru(x)
        x = x.permute(0, 2, 1)

        encs = list()
        for contraction in self.contractions:
            x, enc = contraction(x)
            encs.append(enc)
        x = self.bottleneck(x)
        for expansion, enc in zip(self.expansions, reversed(encs)):
            x = expansion(x, enc)
        
        x = self.unembed(x)
        x = x.squeeze(1)
        if logits:
            return x
        else:
            x = self.sigmoid(x)
            return x

class EncoderLayer(nn.Module):
    def __init__(self, input_channels, output_channels, activation="GELU", num_heads=8, dropout=0.1):
        super(EncoderLayer, self).__init__()
        fn = getattr(nn, activation)
        self.skip = nn.Linear(input_channels, output_channels)
        self.conv = modules.MultiConv1D(input_channels, [4 * output_channels], output_channels, 3, 1, 1, 1, activation, dropout)
        self.norm1 = nn.BatchNorm1d(output_channels)
        self.gru = nn.GRU(input_channels, output_channels, batch_first=True, bidirectional=True)
        self.norm2 = nn.LayerNorm(4 * output_channels)
        self.attn = nn.MultiheadAttention(4 * output_channels, num_heads, batch_first=True, dropout=dropout)
        self.norm3 = nn.LayerNorm(4 * output_channels)
        self.mlp = modules.MLP(4 * output_channels, [16 * output_channels], output_channels, activation)
        self.norm4 = nn.LayerNorm(output_channels)
        self.activation = fn()
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        skip = self.skip(x)
        conv = self.activation(self.norm1(self.conv(x.permute(0, 2, 1))))
        gru, _ = self.gru(x)
        x = torch.cat([conv.permute(0, 2, 1), gru, skip], dim=2)
        x = self.dropout(self.norm2(x))

        res, _ = self.attn(x, x, x)
        x = self.norm3(x + res)
        res = self.mlp(x)
        x = self.norm4(res + skip).permute(0, 2, 1)
        return x

class ContractionLayer(nn.Module):
    def __init__(self, input_channels, output_channels, activation="GELU", num_heads=8, pooling="MaxPool1d", pool_size=2, dropout=0.1):
        super(ContractionLayer, self).__init__()
        pool = getattr(nn, pooling)
        self.encoder = EncoderLayer(input_channels, output_channels, activation, num_heads, dropout)
        self.pool = pool(pool_size)
    
    def forward(self, x):
        enc = self.encoder(x)
        x = self.pool(enc)
        return x, enc
    
class ExpansionLayer(nn.Module):
    def __init__(self, input_channels, output_channels, activation="GELU", pool_size=2, num_heads=8, dropout=0.1):
        super(ExpansionLayer, self).__init__()
        fn = getattr(nn, activation)
        kernel_transpose, padding_transpose = modules.compute_convt(pool_size)
        self.unpool = nn.ConvTranspose1d(input_channels, output_channels, kernel_size=kernel_transpose, stride=pool_size, padding=padding_transpose)
        self.norm = nn.BatchNorm1d(output_channels)
        self.encoder = EncoderLayer(2 * output_channels, output_channels, activation, num_heads, dropout)
        self.activation = fn()

    def forward(self, x, enc):
        x = self.unpool(x)
        x = F.interpolate(x, size=enc.size(2), mode="linear")
        x = self.activation(self.norm(x))
        x = torch.cat([x, enc], dim=1)
        x = self.encoder(x)
        return x
    
if __name__ == "__main__":
    model = H15(16, 3, 7, "GELU", 2, 4, 4, 0.1).to(torch.device("cuda"))
    x = torch.randn(8, 7, 4000).to(torch.device("cuda"))
    y = model(x)
    print(y.shape)
