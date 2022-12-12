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
from models.efficient_net import EfficientNetBasedModel


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

@dataclass
class TrainConfig:
    epoch: int
    batch_size: int
    test_size: float
    model_name: str
    num_classes: int
    on_memory: bool
    image_folder_path: str
    image_meta_data_path: str
    weight_path: str
    save_path: str
    num_workers: int
    cropper_weight_path: str
    cropper_input_size: int
    cropper_output_size: int

    @classmethod
    def from_json(cls, json_object: dict):
        return cls(
            epoch=json_object['epoch'],
            batch_size=json_object['batch_size'],
            test_size=json_object['test_size'],
            model_name=json_object['model_name'],
            num_classes=json_object['num_classes'],
            on_memory=json_object['on_memory'],
            image_folder_path=json_object['image_folder_path'],
            image_meta_data_path=json_object['image_meta_data_path'],
            weight_path=json_object['weight_path'],
            save_path=json_object['save_path'],
            num_workers=json_object['num_workers'],
            cropper_weight_path=json_object['cropper_weight_path'],
            cropper_input_size=json_object['cropper_input_size'],
            cropper_output_size=json_object['cropper_output_size']
        )

@dataclass
class InferenceConfig:
    model_name: str
    weight_path: str
    num_classes: int
    mapping_path: str

    @classmethod
    def from_json(cls, json_object:dict):
        return cls(
            model_name=json_object['model_name'],
            weight_path=json_object['weight_path'],
            num_classes=json_object['num_classes'],
            mapping_path=json_object['mapping_path']
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


def load_image_as_tensor(path: str, resize: int = None, batch_dim: bool = True) -> Tensor:
    img = read_image(path)
    if resize:
        img = Resize((resize, resize))(img)
    img = img.type(FloatTensor) / 255.0
    
    return img.unsqueeze(0) if batch_dim else img



def get_class_prediction_from_logit(class_logit: Tensor) -> List[int]:
    pred = torch.sigmoid(class_logit)
    pred = torch.where(pred >= 0.5, 1.0, 0.0)
    return pred.nonzero(as_tuple=True)[0].tolist()


def make_estimator(
    model_config: ModelConfig,
    num_classes: int = 21,
) -> nn.Module:
        if "swin" in model_config.name:
            return _make_swin_v2_based_estimator(model_config, num_classes)
        elif "efficientnet" in model_config.name:
            return _make_efficient_net_based_estimator(model_config, num_classes)
        else:
            raise NotImplementedError


def _make_swin_v2_based_estimator(
    model_config: ModelConfig,
    num_classes: int = 21,
) -> nn.Module:

    backbone = timm.create_model(model_config.name)
    backbone.head = None

    return SwinV2BasedEstimator(
        backbone=backbone,
        feature_out_size=model_config.feature_out_size,
        linear_hidden_size=model_config.feature_out_size,
        num_classes=num_classes,
    )

def _make_efficient_net_based_estimator(
    model_config: ModelConfig,
    num_classes: int = 21,
) -> nn.Module:

    backbone = timm.create_model(model_config.name)
    backbone.head = None

    return EfficientNetBasedModel(
        backbone=backbone,
        feature_out_size=model_config.feature_out_size,
        linear_hidden_size=model_config.feature_out_size,
        num_classes=num_classes,
    )
    

