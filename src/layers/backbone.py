import torch
import torch.nn as nn
import torchvision.models as models

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1]) 
        self.activation_dim = 2048

    def forward(self, x):
        with torch.no_grad():
            feat = self.feature_extractor(x) 
            feat = feat.view(feat.size(0), -1) 
        return feat
