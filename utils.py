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
    target_rmse: float
    target_acc: float

    @classmethod
    def from_json(cls, json_object: dict):
        return cls(
            epoch=json_object["epoch"],
            batch_size=json_object["batch_size"],
            test_size=json_object["test_size"],
            model_name=json_object["model_name"],
            num_classes=json_object["num_classes"],
            on_memory=json_object["on_memory"],
            image_folder_path=json_object["image_folder_path"],
            image_meta_data_path=json_object["image_meta_data_path"],
            weight_path=json_object["weight_path"],
            save_path=json_object["save_path"],
            num_workers=json_object["num_workers"],
            cropper_weight_path=json_object["cropper_weight_path"],
            cropper_input_size=json_object["cropper_input_size"],
            cropper_output_size=json_object["cropper_output_size"],
            target_rmse=json_object["target_rmse"],
            target_acc=json_object["target_acc"],
        )


@dataclass
class InferenceConfig:
    model_name: str
    weight_path: str
    num_classes: int
    mapping_path: str

    @classmethod
    def from_json(cls, json_object: dict):
        return cls(
            model_name=json_object["model_name"],
            weight_path=json_object["weight_path"],
            num_classes=json_object["num_classes"],
            mapping_path=json_object["mapping_path"],
        )


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


def load_image_as_tensor(
    path: str, resize: int = None, batch_dim: bool = True
) -> Tensor:
    img = read_image(path)
    if resize:
        img = Resize((resize, resize))(img)
    img = img.type(FloatTensor) / 255.0

    return img.unsqueeze(0) if batch_dim else img


def get_class_prediction_from_logit(class_logit: Tensor) -> List[List[int]]:
    """
    input shape: (batch_size, num_classes)
    output: 2d list of class indices
    """
    pred = torch.sigmoid(class_logit)
    pred = torch.where(pred >= 0.5, 1.0, 0.0)
    # 각 배치마다 값이 1인 곳의 인덱스를 가져온다
    return [torch.where(pred[i] == 1.0)[0].tolist() for i in range(pred.shape[0])]


# calculate f1 score for multi-label classification
def f1_score(y_true: Tensor, y_pred: Tensor, epsilon: float = 1e-7) -> float:
    """
    input: class logit tensor. shape of (batch_size, num_classes)
    output: mean f1 score of all classes (float)
    """
    y_pred = torch.sigmoid(y_pred)
    y_pred = torch.where(y_pred >= 0.5, 1.0, 0.0)
    # true positive
    tp = (y_true * y_pred).sum(dim=0)
    # false positive
    fp = ((1 - y_true) * y_pred).sum(dim=0)
    # false negative
    fn = (y_true * (1 - y_pred)).sum(dim=0)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1.mean().item()
