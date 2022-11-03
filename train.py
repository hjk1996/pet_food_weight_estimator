import os
from datetime import datetime
from typing import Any
import json

import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T


from loss_fn import MultiTaskLossWrapper
from dataset import make_dataloaders
from utils import make_swin_v2_based_estimator, save_model_weights


def train_one_epoch(
    epoch: int,
    model: torch.nn.Module,
    dataloader: DataLoader,
    writer: SummaryWriter,
    loss_fn: MultiTaskLossWrapper,
    optimizer: Any,
):
    running_loss = 0.0
    running_mae = 0.0
    dataloader_len = len(dataloader)

    for weight, food_type, img in dataloader:
        optimizer.zero_grad()
        preds = model(img)
        loss = loss_fn(preds, (weight, food_type))
        mae = loss_fn.mae(preds[0], weight)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_mae += mae.item()

    epoch_loss = running_loss / dataloader_len
    epoch_mae = running_mae / dataloader_len

    print(f"EPOCH[{epoch}] Train/Loss: {epoch_loss}")
    print(f"EPOCH[{epoch}] Train/MAE: {epoch_mae}")

    writer.add_scalar("Loss/train/total", epoch_loss, epoch)
    writer.add_scalar("Loss/train/mae", epoch_mae, epoch)


def evaluate_classification(gt: torch.Tensor, pred_logit: torch.Tensor) -> int:
    pred = torch.sigmoid(pred_logit)
    pred = torch.where(pred >= 0.5, 1.0, 0.0)
    return torch.all(gt == pred, dim=1).sum().item()


def validate_one_epoch(
    epoch: int,
    model: torch.nn.Module,
    dataloader: DataLoader,
    writer: SummaryWriter,
    loss_fn: MultiTaskLossWrapper,
    classification: bool
) -> float:
    running_loss = 0.0
    running_mae = 0.0
    if classification:
        right_count = 0
    dataloader_len = len(dataloader)

    for weight, food_type, img in dataloader:
        with torch.no_grad():
            preds = model(img)
            loss = loss_fn(preds, (weight, food_type))
            running_loss += loss.item()
            running_mae += loss_fn.mae(preds[0], weight)
            if classification:
                right_count += evaluate_classification(food_type, preds[1])

    epoch_loss = running_loss / dataloader_len
    epoch_mae = running_mae / dataloader_len
    if classification:
        epoch_acc = right_count / dataloader_len

    print(f"EPOCH[{epoch}] Val/Loss: {epoch_loss}")
    print(f"EPOCH[{epoch}] Val/MAE: {epoch_mae}")
    if classification:
        print(f"EPOCH[{epoch}] Val/ACC: {epoch_acc}", "\n")

    writer.add_scalar("Loss/valid/total", epoch_loss, epoch)
    writer.add_scalar("Loss/valid/mae", epoch_mae, epoch)
    if classification:
        writer.add_scalar("Acc/valid", epoch_acc, epoch)

    return epoch_mae


def train_and_valid(
    model: torch.nn.Module,
    dataloaders: dict,
    n_epochs: int,
    save_path: str,
    classification: bool = True,
) -> SummaryWriter:
    loss_fn = MultiTaskLossWrapper(classification=classification)
    optimizer = torch.optim.Adam(model.parameters())
    best_mae = float("inf")
    best_weights = model.state_dict()
    save_path = os.path.join(save_path, str(datetime.now()).replace(":", "-").replace(" ", "_").split('.')[0])
    writer = SummaryWriter(os.path.join(save_path, "log"))

    for i in range(n_epochs):
        for phase in ["train", "test"]:

            dataloader = dataloaders[phase]

            if phase == "train":
                model.train()
                train_one_epoch(
                    epoch=i,
                    model=model,
                    dataloader=dataloader,
                    writer=writer,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                )
            else:
                model.eval()
                epoch_mae = validate_one_epoch(
                    epoch=i,
                    model=model,
                    dataloader=dataloader,
                    writer=writer,
                    loss_fn=loss_fn,
                    classification=classification
                )

                if epoch_mae < best_mae:
                    best_mae = epoch_mae
                    best_weights = model.state_dict()
                    save_model_weights(model.state_dict(), save_path, best=True)

    writer.close()
    save_model_weights(best_weights, save_path, best=False)

    return writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=1000, type=int, help="훈련 에포크 수")
    parser.add_argument("--batch_size", default=32, type=int, help="배치 사이즈")
    parser.add_argument("--hidden_size", default=768, type=int, help="FC 유닛 수")
    parser.add_argument(
        "--classification", default=True, type=bool, help="사료 종류 분류할지 여부"
    )
    parser.add_argument("--n_classes", default=21, type=int, help="사료 클래스 수")
    parser.add_argument("--test_size", default=0.2, type=float, help="테스트 데이터셋 비율")
    parser.add_argument("--weights", default=None, type=str, help="사용할 모델 가중치의 경로")
    args = parser.parse_args()

    with open("config.json", "r") as f:
        config = json.load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloaders = make_dataloaders(
        meta_data_path=config["image_meta_data_path"],
        img_dir=config["image_folder_path"],
        n_classes=args.n_classes,
        device=device,
        test_size=args.test_size,
        batch_size=args.batch_size,
        transform=T.AugMix(),
    )

    model = make_swin_v2_based_estimator(
        device=device,
        linear_hidden_size=args.hidden_size,
        classification=args.classification,
        n_classes=args.n_classes,
    )

    if args.weights:
        state_dict = torch.load(args.weights)
        model.load_state_dict(state_dict)

    train_and_valid(
        model=model,
        dataloaders=dataloaders,
        n_epochs=args.epoch,
        save_path=config["save_path"],
    )
