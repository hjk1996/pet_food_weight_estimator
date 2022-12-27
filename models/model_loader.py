import json
from dataclasses import dataclass


import torch.nn as nn

from models.model import Model
from models.adaptors import adaptor_mapper


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
    with open("model_configs.json", "r") as f:
        return ModelConfig.from_json(json.load(f)[model_name])


def make_model(model_name: str, num_classes: int, hidden_size: int = 512) -> nn.Module:
    config = load_model_config(model_name)
    adatpor_class = adaptor_mapper.get(config.name, None)
    if adatpor_class is None:
        raise ValueError(f"adaptor for {config.name} is not defined.")
    adatpor = adatpor_class(
        feature_out_size=config.feature_out_size, hidden_size=hidden_size
    )
    return Model(
            backbone=config.name,
            adaptor=adatpor,
            num_classes=num_classes,
            hidden_size=hidden_size,
        )
