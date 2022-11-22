import os
from typing import List

import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor, FloatTensor
from torchvision.io import read_image
from torchvision.transforms import Resize
import timm

from models import SwinV2BasedEstimator


def save_model_weights(weights: dict, save_path: str, best: bool = False) -> None:
    new_save_path = os.path.join(save_path, "best.pt" if best else "last.pt")
    torch.save(weights, new_save_path)


def indice_to_name_mapper(file_path: str) -> dict:
    df = pd.read_csv(file_path)
    indices = df["indice"].to_list()
    names = df["name"].to_list()
    return {indice: name for indice, name in zip(indices, names)}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_image_as_input_tensor(path: str, resize: int = None) -> Tensor:
    img = read_image(path)
    if resize:
        img = Resize(resize)(img)
    return img.type(FloatTensor) / 255.0


def get_class_prediction_from_logit(class_logit: Tensor) -> List[int]:
    pred = torch.sigmoid(class_logit)
    pred = torch.where(pred >= 0.5, 1.0, 0.0)
    return pred.nonzero(as_tuple=True)[0].tolist()


def make_swin_v2_based_estimator(
    device: torch.device,
    linear_hidden_size: int = 768,
    classification: bool = True,
    n_classes: int = 21,
) -> nn.Module:
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
