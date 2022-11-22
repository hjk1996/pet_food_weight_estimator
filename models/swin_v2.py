from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torchvision.transforms import Resize
import timm


class SwinV2BasedEstimator(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_out_size: int,
        linear_hidden_size: int,
        detector: nn.Module = None,
        cropping: bool = False,
        num_classes: int = 0,
    ):
        super().__init__()

        assert (detector == None and cropping == False) or (detector == True)

        self.backbone = backbone
        self.detector = detector
        self.resizer = Resize((256, 256))

        self.cropping = cropping

        self.feature_out_size = feature_out_size
        self.linear_hidden_size = linear_hidden_size
        self.num_classes = num_classes

        self.estimator = nn.Sequential(
            self._make_adaptor_linear_block(),
            self._make_linear_block(),
            self._make_last_linear_block(out_features=1)
        )

        self.classifier = nn.Sequential(
            self._make_adaptor_linear_block(),
            self._make_linear_block(),
            self._make_last_linear_block(out_features=num_classes)
        )


        self.relu = nn.ReLU()

    def _make_adaptor_linear_block(self) -> nn.modules.container.Sequential:
        return nn.Sequential(
            nn.Linear(
                in_features=self.feature_out_size, out_features=self.linear_hidden_size
            ),
            nn.BatchNorm1d(num_features=self.linear_hidden_size),
            nn.ReLU(),
        )

    def _make_linear_block(self) -> nn.modules.container.Sequential:
        return nn.Sequential(
            nn.Linear(
                in_features=self.linear_hidden_size,
                out_features=self.linear_hidden_size,
            ),
            nn.BatchNorm1d(num_features=self.linear_hidden_size),
            nn.ReLU(),
        )

    def _make_last_linear_block(
        self, out_features: int
    ) -> nn.modules.container.Sequential:
        return nn.Linear(
            in_features=self.linear_hidden_size, out_features=out_features
        )

    def forward_feature(self, image) -> Tensor:
        if self.cropping:
            image = self.detector(image)
            image = self.resizer(image)

        feature_map = self.backbone.forward_features(image)
        feature_map = feature_map.mean(1)
        return feature_map

    def forward(self, image) -> Tuple[Tensor, Tensor]:
        feature_map = self.forward_feature(image)
        weight = self.estimator(feature_map)
        class_logit = self.classifier(feature_map)
        return weight, class_logit
