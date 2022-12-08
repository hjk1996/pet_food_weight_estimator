import torch
from torch import Tensor, FloatTensor
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 손실값 계산할 때 평균값으로 계산함.
        self.mse = nn.MSELoss()

    def forward(self, yhat, y) -> Tensor:
        return torch.sqrt(self.mse(yhat, y))


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, classification: bool = True):
        super(MultiTaskLossWrapper, self).__init__()
        self.weight_loss_fn = RMSELoss()
        self.class_loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        # self.mae = nn.L1Loss()
        self.classification = classification

    def multi_task_forward(self, preds, gts):
        weight_loss = self.weight_loss_fn(preds[0], gts[0])
        class_loss = self.class_loss_fn(preds[1], gts[1])
        return (weight_loss + class_loss).type(FloatTensor)

    def single_task_forward(self, preds, gts):
        return self.weight_loss_fn(preds[0], gts[0])

    def forward(self, preds, gts) -> Tensor:
        return (
            self.multi_task_forward(preds, gts)
            if self.classification
            else self.single_task_forward(preds, gts)
        )
