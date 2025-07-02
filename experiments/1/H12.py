import torch
import torch.nn as nn
import modules

class H12(nn.Module):
    def __init__(self, d_model, input_size, max_len, encoder_layers, decoder_layers, num_heads, activation, dropout):
        super(H12, self).__init__()
        fn = getattr(nn, activation)
        self.embed = modules.Embedding1D(input_size, d_model, "MLP", [4 * d_model], activation, encoding="pos", max_len=max_len)
        self.transformer = nn.Transformer(d_model, num_heads, encoder_layers, decoder_layers, 4 * d_model, activation=fn(), batch_first=True, dropout=dropout)
        self.unembed = modules.MLP(d_model, [4 * d_model], 1, activation)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, logits=False):
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        x = self.transformer(x, x)
        x = self.unembed(x)
        x = x.squeeze(2)
        if logits:
            return x
        else:
            x = self.sigmoid(x)
            return x
    
if __name__ == "__main__":
    model = H12(64, 7, 4096, 4, 4, 4, "GELU", 0.1).to(torch.device("cuda"))
    x = torch.randn(16, 7, 1000).to(torch.device("cuda"))
    y = model(x)
    print(y.shape)
