import os
import json
from typing import List
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor, FloatTensor
from torchvision.io import read_image
from torchvision.transforms import Resize
import timm

from models.swin_v2 import SwinV2BasedEstimator


@dataclass
class ModelConfig:
    name: str
    resolution: int
    feature_out_size: int

    @classmethod
    def from_json(cls, json_object: dict):
        return cls(
            name=json_object["name"],
            resolution=json_object["resolution"],
            feature_out_size=json_object["feature_out_size"],
        )


def load_model_config(model_name: str) -> ModelConfig:
    with open('model_configs.json', 'r') as f:
        return ModelConfig.from_json(json.load(f)[model_name])

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


def load_image_as_tensor(path: str, resize: int = None) -> Tensor:
    img = read_image(path)
    if resize:
        img = Resize((resize, resize))(img)
    img = img.type(FloatTensor) / 255.0
    img = img.unsqueeze(0)
    return img


def get_class_prediction_from_logit(class_logit: Tensor) -> List[int]:
    pred = torch.sigmoid(class_logit)
    pred = torch.where(pred >= 0.5, 1.0, 0.0)
    return pred.nonzero(as_tuple=True)[0].tolist()


def make_swin_v2_based_estimator(
    device: torch.device,
    model_config: ModelConfig,
    num_classes: int = 21,
) -> nn.Module:

    backbone = timm.create_model(model_config.name)
    backbone.head = None

    model = SwinV2BasedEstimator(
        backbone=backbone,
        feature_out_size=model_config.feature_out_size,
        linear_hidden_size=model_config.feature_out_size,
        num_classes=num_classes,
    ).to(device)
    return model
