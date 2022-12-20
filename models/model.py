import torch.nn as nn
import timm
from models.head import Head

class Model(nn.Module):

    def __init__(self, backbone: str, adaptor: nn.Module, hidden_size: int, num_classes: int = 0):
        '''
        backbone: timm model name.
        adaptor: transform the output of the backbone to the input of the head.
        head: head of the model. it does the classification and regression.
        '''
        super().__init__()
        
        self.backbone = timm.create_model(backbone)
        self.adaptor = adaptor
        self.head = Head(num_classes=num_classes, hidden_size=hidden_size)
    
    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.adaptor(x)
        x = self.head(x)
        return x