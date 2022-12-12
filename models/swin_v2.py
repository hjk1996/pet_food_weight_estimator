from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn


class SwinV2BasedEstimator(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_out_size: int,
        linear_hidden_size: int,
        num_classes: int = 0,
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_out_size = feature_out_size
        self.linear_hidden_size = linear_hidden_size
        self.num_classes = num_classes

        self.estimator = nn.Sequential(
            self._make_adaptor_linear_block(),
            self._make_linear_block(),
            self._make_last_linear_block(out_features=1),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            self._make_adaptor_linear_block(),
            self._make_linear_block(),
            self._make_last_linear_block(out_features=num_classes)
        )



    def _make_adaptor_linear_block(self) -> nn.modules.container.Sequential:
        return nn.Sequential(
            nn.Linear(
                in_features=self.feature_out_size, out_features=self.linear_hidden_size
            ),
            nn.BatchNorm1d(num_features=self.linear_hidden_size),
            nn.SiLU(),
        )

    def _make_linear_block(self) -> nn.modules.container.Sequential:
        return nn.Sequential(
            nn.Linear(
                in_features=self.linear_hidden_size,
                out_features=self.linear_hidden_size,
            ),
            nn.BatchNorm1d(num_features=self.linear_hidden_size),
            nn.SiLU(),
        )

    def _make_last_linear_block(
        self, out_features: int
    ) -> nn.modules.container.Sequential:
        return nn.Linear(
            in_features=self.linear_hidden_size, out_features=out_features
        )
    
        
    def forward_feature(self, x: Tensor) -> Tensor:
        feature_map = self.backbone.forward_features(x)
        feature_map = feature_map.mean(1)
        feature_map = self.droupout(feature_map)
        return feature_map


    def forward(self, x) -> Tuple[Tensor, Tensor]:
        '''
            return:
                weight [batch_size, 1]
                class_logit [batch_size,  n_classes]
        '''
        feature_map = self.forward_feature(x)
        weight = self.estimator(feature_map)
        class_logit = self.classifier(feature_map)
        return weight, class_logit
