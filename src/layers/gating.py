import torch
import torch.nn.functional as F

class Gating(torch.nn.Module):
    def __init__(self, offset=0.1):
        super().__init__()
        self.offset = offset

    def forward(self, proj):
        return F.relu(proj - self.offset)   
