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
    def __init__(self, classification: bool = True, alpha: float = 0.8, beta: float = 100):
        super(MultiTaskLossWrapper, self).__init__()
        self.weight_loss_fn = RMSELoss()
        self.class_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        # self.mae = nn.L1Loss()
        self.classification = classification
        self.last_loss = 0
        self.last_rmse = 0
        self.last_bce = 0
        self.alpha = alpha
        self.beta = beta

    def multi_task_forward(self, preds, gts):
        weight_loss = self.weight_loss_fn(preds[0], gts[0])
        self.last_rmse = weight_loss.item()
        class_loss = self.class_loss_fn(preds[1], gts[1])
        self.last_bce = class_loss.item()
        loss = self.alpha * weight_loss + self.beta * (1 - self.alpha) * class_loss
        self.last_loss = loss.item()
        return loss

    def single_task_forward(self, preds, gts):
        loss = self.weight_loss_fn(preds[0], gts[0]).item()
        self.last_loss = loss.item()
        return loss

    def forward(self, preds, gts) -> Tensor:
        return (
            self.multi_task_forward(preds, gts)
            if self.classification
            else self.single_task_forward(preds, gts)
        )
