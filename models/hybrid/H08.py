import torch
import torch.nn as nn
import modules

class H08(nn.Module):
    def __init__(self, d_model, num_layers, input_size, activation, dropout):
        super(H08, self).__init__()
        self.embed = modules.Embedding1D(input_size, d_model, "MLP", [4 * d_model], activation)
        self.gru = nn.GRU(d_model, d_model, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.unembed = modules.MLP(3 * d_model, [12 * d_model], 1, activation)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, logits=False):
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        gru, _ = self.gru(x)
        x = torch.cat([x, gru], dim=2)
        x = self.unembed(x)
        x = x.squeeze(2)
        if logits:
            return x
        else:
            x = self.sigmoid(x)
            return x
    
if __name__ == "__main__":
    model = H08(256, 3, 8, "GELU", 0.1).to(torch.device("cuda"))
    x = torch.randn(16, 8, 1000).to(torch.device("cuda"))
    y = model(x)
    print(y.shape)
