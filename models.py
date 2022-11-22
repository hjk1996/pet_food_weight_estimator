from typing import Tuple

from torch import Tensor
import torch.nn as nn
from torchvision.transforms import Resize


class SwinV2BasedEstimator(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        backbone_out_size: int,
        linear_hidden_size: int,
        detector: nn.Module = None,
        cropping: bool = False,
        classification: bool = False,
        num_classes: int = 0,
    ):
        super().__init__()

        assert (classification == True and num_classes != 0) or (
            classification == False and num_classes == 0
        )
        assert (detector == None and cropping == False) or (detector == True)

        self.backbone = backbone
        self.detector = detector
        self.resizer = Resize((256, 256))

        self.classification = classification
        self.cropping = cropping

        self.backbone_out_size = backbone_out_size
        self.linear_hidden_size = linear_hidden_size
        self.num_classes = num_classes

        self.estimator_block_1 = self._make_adaptor_linear_block()
        self.estimator_block_2 = self._make_linear_block()
        self.estimator_block_3 = self._make_last_linear_block(last_out_feature=1)

        if classification:
            self.classifier_block_1 = self._make_adaptor_linear_block()
            self.classifier_block_2 = self._make_linear_block()
            self.classifier_block_3 = self._make_last_linear_block(
                last_out_feature=num_classes
            )

        self.relu = nn.ReLU()

    def _make_adaptor_linear_block(self) -> nn.modules.container.Sequential:
        return nn.Sequential(
            nn.Linear(
                in_features=self.backbone_out_size, out_features=self.linear_hidden_size
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
        self, last_out_feature: int
    ) -> nn.modules.container.Sequential:
        return nn.Linear(
            in_features=self.linear_hidden_size, out_features=last_out_feature
        )

    def _forward_with_classifier(self, feature_map: Tensor):
        x1 = self.estimator_block_1(feature_map)
        x1 = self.estimator_block_2(x1)
        x1 = self.estimator_block_3(x1)
        weight = self.relu(x1)

        x2 = self.classifier_block_1(feature_map)
        x2 = self.classifier_block_2(x2)
        class_logit = self.classifier_block_3(x2)

        return weight, class_logit

    def _forward_without_classifier(self, feature_map: Tensor):
        x1 = self.estimator_block_1(feature_map)
        x1 = self.estimator_block_2(x1)
        x1 = self.estimator_block_3(x1)
        weight = self.relu(x1)

        return weight

    def forward(self, image) -> Tuple[Tensor, Tensor]:
        if self.cropping:
            image = self.detector(image)
            image = self.resizer(image)

        feature_map = self.backbone.forward_features(image)
        feature_map = feature_map.mean(1)

        return (
            self._forward_with_classifier(feature_map)
            if self.classification
            else self._forward_without_classifier(feature_map)
        )
