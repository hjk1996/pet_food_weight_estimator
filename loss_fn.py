import torch


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, yhat, y) -> torch.Tensor:
        return torch.sqrt(self.mse(yhat, y))


class MultiTaskLossWrapper(torch.nn.Module):
    def __init__(self, classification: bool = True):
        super(MultiTaskLossWrapper, self).__init__()
        self.weight_loss_fn = RMSELoss()
        self.class_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
        self.mae = torch.nn.L1Loss()
        self.classification = classification

    def multi_task_forward(self, preds, gts):
        weight_loss = self.weight_loss_fn(preds[0], gts[0])
        class_loss = self.class_loss_fn(preds[1], gts[1])
        return (weight_loss + class_loss).type(torch.FloatTensor)

    def single_task_forward(self, preds, gts):
        return self.weight_loss_fn(preds[0], gts[0])

    def forward(self, preds, gts) -> torch.Tensor:
        return (
            self.multi_task_forward(preds, gts)
            if self.classification
            else self.single_task_forward(preds, gts)
        )
