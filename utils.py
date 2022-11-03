import os

import pandas as pd
import torch
import timm

from models import SwinV2BasedEstimator


def save_model_weights(weights: dict, save_path: str, best: bool = False) -> None:
    new_save_path = os.path.join(save_path, "best.pt" if best else "last.pt")
    torch.save(weights, new_save_path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_swin_v2_based_estimator(
    device: torch.device,  linear_hidden_size: int = 768, classification: bool = True, n_classes: int = 21
) -> torch.nn.Module:
    backbone = timm.create_model("swinv2_tiny_window8_256")
    backbone.head = None
    model = SwinV2BasedEstimator(
        backbone=backbone,
        backbone_out_size=768,
        linear_hidden_size=linear_hidden_size,
        classification=classification,
        num_classes=n_classes,
    ).to(device)
    return model
