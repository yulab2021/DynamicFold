import ViennaRNA
import numpy as np
import torch
import torch.nn.functional as F

class RNAFold:
    def __init__(self, reactivity=True, m=1.9, b=-0.7):
        self.device = torch.device("cpu")
        self.reactivity = reactivity
        self.m = m
        self.b = b

    def move(self, device):
        self.device = device
        return self
    
    def __call__(self, sequences, reactivities):
        outputs = list()
        with torch.no_grad():
            if self.reactivity:
                max_len = max([len(sequence) for sequence in sequences])
                for sequence in sequences:
                    fc = ViennaRNA.fold_compound(sequence)
                    fc.pf()
                    output = torch.tensor(np.array(fc.bpp())[1:,1:]).unsqueeze(0)
                    output = F.pad(output, (0, max_len - output.size(1), 0, max_len - output.size(2))).clamp(min=0.0, max=1.0)
                    outputs.append(output)
            else:
                assert len(sequences) == len(reactivities)
                max_len = max([len(sequence) for sequence in sequences])
                for sequence, reactivity in zip(sequences, reactivities):
                    assert len(sequence) == len(reactivity)
                    fc = ViennaRNA.fold_compound(sequence)
                    fc.sc_add_SHAPE_deigan(reactivity, self.m, self.b)
                    fc.pf()
                    output = np.array(fc.bpp())[1:,1:]
                    output = F.pad(output, (0, max_len - output.size(1), 0, max_len - output.size(2))).clamp(min=0.0, max=1.0)
                    outputs.append(output)
            outputs = torch.tensor(np.array(outputs), dtype=torch.float32, device=self.device)
        return outputs
