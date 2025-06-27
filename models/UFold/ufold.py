import torch
import torch.nn.functional as F
from Network import U_Net
from processing import postprocess_new_nc as postprocess
from processing import creatmat

class UFold:
    def __init__(self, img_ch, checkpoint):
        self.device = torch.device("cpu")
        self.pairings = [[i, j] for i in range(4) for j in range(4)]
        self.tokens = {'A': [1.0, 0, 0, 0], 'U': [0, 1.0, 0, 0], 'C': [0, 0, 1.0, 0], 'G': [0, 0, 0, 1.0], 'N': [1.0, 1.0, 1.0, 1.0]}

        self.model = U_Net(img_ch=img_ch)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.model.eval()

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self
        
    def embed(self, sequence):
        encoding = torch.tensor([self.tokens[nt] for nt in sequence], device=self.device, dtype=torch.float32)
        embedding = list()
        for i, j in self.pairings:
            embedding.append(torch.matmul(encoding[:, i].reshape(-1, 1), encoding[:, j].reshape(1, -1)))
        embedding.append(creatmat(encoding, self.device))
        embedding = torch.stack(embedding).unsqueeze(0)
        return embedding, encoding.unsqueeze(0)
    
    def __call__(self, sequences):
        outputs = list()
        max_len = max([len(sequence) for sequence in sequences])
        with torch.no_grad():
            for sequence in sequences:
                embedding, encoding = self.embed(sequence)
                utility = self.model(embedding)
                output = postprocess(utility, encoding, 0.0001, 0.1, 100, 1.6, True, 1.5, 1, self.device)
                output = F.pad(output, (0, max_len - output.size(1), 0, max_len - output.size(2))).clamp(min=0.0, max=1.0)
                outputs.append(output)
            outputs = torch.cat(outputs, dim=0)
        return outputs
