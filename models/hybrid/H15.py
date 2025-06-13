import torch
import torch.nn as nn
import torch.nn.functional as F
import modules

class H15(nn.Module):
    def __init__(self, d_model, num_layers, input_size, activation, channel_rate, pool_size, num_heads, dropout):
        super(H15, self).__init__()
        self.embed = modules.Embedding1D(input_size, d_model, "conv", [4 * d_model], activation)
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
        x = self.embed(x)
        x, _ = self.gru(x.permute(0, 2, 1))
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
        self.gru = nn.GRU(input_channels, output_channels, batch_first=True, bidirectional=True)
        self.norm1 = nn.LayerNorm(4 * output_channels)
        self.attn = nn.TransformerEncoderLayer(4 * output_channels, num_heads, 16 * output_channels, activation=fn(), batch_first=True, dropout=dropout)
        self.mlp = modules.MLP(4 * output_channels, [16 * output_channels], output_channels, activation)
        self.norm2 = nn.LayerNorm(output_channels)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        skip = self.skip(x)
        conv = self.conv(x.permute(0, 2, 1))
        gru, _ = self.gru(x)
        x = torch.cat([conv.permute(0, 2, 1), gru, skip], dim=2)
        x = self.dropout(self.norm1(x))
        x = self.attn(x)
        res = self.mlp(x)
        x = self.norm2(res + skip).permute(0, 2, 1)
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
        self.attn = nn.TransformerDecoderLayer(output_channels, num_heads, 4 * output_channels, activation=fn(), batch_first=True, dropout=dropout)
        self.encoder = EncoderLayer(2 * output_channels, output_channels, activation, num_heads, dropout)

    def forward(self, x, enc):
        x = self.unpool(x)
        x = F.interpolate(x, size=enc.size(2), mode="linear")
        enc = self.attn(enc.permute(0, 2, 1), x.permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.cat([x, enc], dim=1)
        x = self.encoder(x)
        return x
    
if __name__ == "__main__":
    model = H15(16, 4, 8, "GELU", 2, 2, 8, 0.1).to(torch.device("cuda"))
    x = torch.randn(16, 8, 1000).to(torch.device("cuda"))
    y = model(x)
    print(y.shape)
