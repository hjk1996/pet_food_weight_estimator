import os
from datetime import datetime
from typing import Any
import json

import argparse
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T

from loss_fn import MultiTaskLossWrapper
from dataset import make_dataloaders
from utils import save_model_weights, load_model_config, make_swin_v2_based_estimator,TrainConfig



def train_one_epoch(
    epoch: int,
    model: nn.Module,
    dataloader: DataLoader,
    writer: SummaryWriter,
    loss_fn: MultiTaskLossWrapper,
    optimizer: Any,
):
    running_loss = 0.0
    dataloader_len = len(dataloader)

    for gram, food_type, img in dataloader:
        optimizer.zero_grad()
        preds = model(img)
        loss = loss_fn(preds, (gram, food_type))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / dataloader_len
    print(f"EPOCH[{epoch}] Train/Loss: {epoch_loss}")
    writer.add_scalar("Train/total_loss", epoch_loss, epoch)


def evaluate_classification(gt: Tensor, pred_logit: Tensor) -> int:
    pred = torch.sigmoid(pred_logit)
    pred = torch.where(pred >= 0.5, 1.0, 0.0)
    return torch.all(gt == pred, dim=1).sum().item()




def validate_one_epoch(
    epoch: int,
    model: nn.Module,
    dataloader: DataLoader,
    writer: SummaryWriter,
    loss_fn: MultiTaskLossWrapper,
) -> float:
    running_loss = 0.0
    running_rmse = 0.0
    right_count = 0
    dataloader_len = len(dataloader)

    for gram, food_type, img in dataloader:
        with torch.no_grad():
            preds = model(img)
            loss = loss_fn(preds, (gram, food_type))
            running_loss += loss.item()
            running_rmse += loss_fn.weight_loss_fn(preds[0], gram)
            # 왜 right_count가 제대로 집계되지 않는지 확인해야함.
            right_count += evaluate_classification(food_type, preds[1])

    epoch_loss = running_loss / dataloader_len
    epoch_rmse = running_rmse / dataloader_len
    epoch_acc = right_count / dataloader_len

    print(f"EPOCH[{epoch}] Val/Loss: {epoch_loss}")
    print(f"EPOCH[{epoch}] Val/RMSE: {epoch_rmse}")
    print(f"EPOCH[{epoch}] Val/ACC: {epoch_acc}", "\n")

    writer.add_scalar("Valid/total_loss", epoch_loss, epoch)
    writer.add_scalar("Valid/rmse", epoch_rmse, epoch)
    writer.add_scalar("Valid/acc", epoch_acc, epoch)

    return epoch_rmse


def train_and_valid(
    model: nn.Module,
    dataloaders: dict,
    n_epochs: int,
    save_path: str,
) -> SummaryWriter:
    loss_fn = MultiTaskLossWrapper(classification=True)
    optimizer = torch.optim.Adam(model.parameters())
    best_mae = float("inf")
    best_weights = model.state_dict()
    save_path = os.path.join(
        save_path, str(datetime.now()).replace(":", "-").replace(" ", "_").split(".")[0]
    )
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
    parser.add_argument("--train_config_path", type=str, required=True,  help="훈련에 사용할 설정이 정의되어 있는 json file의 경로")
    parser.add_argument('--test_mode', action="store_true")
    args = parser.parse_args()

    with open(args.train_config_path, "r") as f:
        config = json.load(f)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_config = load_model_config(config["model_name"])

    if config.get("cropper_weight_path") and config.get("cropper_input_size") and config.get("cropper_output_size"):
        dataloaders = make_dataloaders(
            image_meta_data_path=config["image_meta_data_path"],
            img_dir=config["image_folder_path"],
            num_classes=config["num_classes"],
            device=device,
            on_memory=config["on_memory"],
            test_size=config["test_size"],
            batch_size=config["batch_size"],
            num_workers=config['num_workers'],
            transform=T.AugMix(),
            cropper_weight_path=config.get("cropper_weight_path"),
            cropper_input_size=config.get("cropper_input_size"),
            cropper_output_size=config.get("cropper_output_size"),
            test_mode=args.test_mode

        )       
    else:


        dataloaders = make_dataloaders(
            image_meta_data_path=config["image_meta_data_path"],
            img_dir=config["image_folder_path"],
            num_classes=config["num_classes"],
            device=device,
            on_memory=config["on_memory"],
            test_size=config["test_size"],
            batch_size=config["batch_size"],
            num_workers=config['num_workers'],
            transform=T.AugMix(),
            test_mode=args.test_mode
        )

    model = make_swin_v2_based_estimator(
        device=device,
        model_config=model_config,
        num_classes=config["num_classes"],
    )

    if config.get("weight_path"):
        model.load_state_dict(torch.load(config.get("weight_path")))

    train_and_valid(
        model=model,
        dataloaders=dataloaders,
        n_epochs=config["epoch"],
        save_path=config["save_path"],
    )
