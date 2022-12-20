import torch
import torch.nn as nn


__all__ = ['adaptor_mapper']

class EfficientNetAdaptor(nn.Module):
    def __init__(self, feature_out_size: int, hidden_size: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(in_features=feature_out_size, out_features=hidden_size)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ap(x).squeeze(-1).squeeze(-1)
        return self.linear(x)

class SwinV2Adaptor(nn.Module):
    def __init__(self, feature_out_size: int, hidden_size: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(in_features=feature_out_size, out_features=hidden_size)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(1)
        return self.linear(x)


adaptor_mapper = {
    'efficientnetv2_l': EfficientNetAdaptor,
    'efficientnetv2_s': EfficientNetAdaptor,
    'swinv2_tiny_window8_256': SwinV2Adaptor,
    'swinv2_small_window8_256': SwinV2Adaptor,
    'swinv2_base_window8_256': SwinV2Adaptor,
}