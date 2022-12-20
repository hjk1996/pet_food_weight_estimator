import torch
import torch.nn as nn

class Head(nn.Module):
    def __init__(self, num_classes: int,  hidden_size: int = 512,):
        super().__init__()
        self.dropout = nn.Dropout(p=0.2)
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.estimator_base_1 = self._make_base()
        self.estimator_base_2 = self._make_base()
        self.estimator = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=1),
            nn.ReLU()
        )
        self.classifier_base_1 = self._make_base()
        self.classifier_base_2 = self._make_base()
        self.classifier = nn.Linear(in_features=self.hidden_size, out_features=num_classes)

    def _make_base(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.SiLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x_1 = self.estimator_base_1(x)
        x_2 = self.classifier_base_2(x)
        x_1 = self.estimator_base_2(x_1 + x_2)
        x_2 = self.classifier_base_1(x_2)
        weight = self.estimator(x_1 + x_2)
        class_logit = self.classifier(x_2)
        return weight, class_logit
    
