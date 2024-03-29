import os
from datetime import datetime
from typing import Any, Tuple, Optional
import json
import time
import argparse

import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from loss_fn import MultiTaskLossWrapper
from models.model_loader import make_model
from dataset import make_dataloaders
from utils import save_model_weights, TrainConfig, f1_score
from augmentation import kim_aug


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    dataloader: DataLoader,
    writer: SummaryWriter,
    loss_fn: MultiTaskLossWrapper,
    optimizer: Any,
    device: torch.device,
):
    running_loss = 0.0
    dataloader_len = len(dataloader)

    for gram, food_type, img in dataloader:
        gram, food_type, img = gram.to(device), food_type.to(device), img.to(device)
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
    """
    input: gt (batch_size, num_classes), pred_logit (batch_size, num_classes)
    output: integer value (number of correct predictions)
    """
    pred = torch.sigmoid(pred_logit)
    pred = torch.where(pred >= 0.5, 1.0, 0.0)
    return torch.all(gt == pred, dim=1).sum().item()


def validate_one_epoch(
    epoch: int,
    model: nn.Module,
    dataloader: DataLoader,
    writer: SummaryWriter,
    loss_fn: MultiTaskLossWrapper,
    device: torch.device,
) -> Tuple[float, float, float]:

    running_loss = 0.0
    running_rmse = 0.0
    right_count = 0
    data_length = 0
    dataloader_len = len(dataloader)
    food_type_gts = []
    food_type_preds = []

    for gram, food_type, img in dataloader:
        gram, food_type, img = gram.to(device), food_type.to(device), img.to(device)
        food_type_gts.append(food_type)
        data_length += gram.shape[0]
        with torch.no_grad():
            preds = model(img)
            food_type_preds.append(preds[1])
            loss = loss_fn(preds, (gram, food_type))
            running_loss += loss_fn.last_loss
            running_rmse += loss_fn.last_rmse
            right_count += evaluate_classification(food_type, preds[1])

    food_type_gts = torch.cat(food_type_gts, dim=0)
    food_type_preds = torch.cat(food_type_preds, dim=0)

    epoch_loss = running_loss / dataloader_len
    epoch_rmse = running_rmse / dataloader_len
    epoch_acc = right_count / data_length
    epoch_f1 = f1_score(food_type_gts, food_type_preds)

    print(f"EPOCH[{epoch}] Val/Loss: {epoch_loss}")
    print(f"EPOCH[{epoch}] Val/RMSE: {epoch_rmse}")
    print(f"EPOCH[{epoch}] Val/ACC: {epoch_acc}")
    print(f"EPOCH[{epoch}] Val/F1: {epoch_f1}")

    writer.add_scalar("Valid/total_loss", epoch_loss, epoch)
    writer.add_scalar("Valid/rmse", epoch_rmse, epoch)
    writer.add_scalar("Valid/acc", epoch_acc, epoch)
    writer.add_scalar("Valid/f1", epoch_f1, epoch)

    return epoch_rmse, epoch_acc, epoch_f1


def train_and_valid(
    model: nn.Module,
    dataloaders: dict,
    n_epochs: int,
    device: torch.device,
    save_path: str,
    target_rmse: float = 0.0,
    targer_acc: float = 1.0,
    targer_f1: float = 1.0,

) -> pd.DataFrame:
    loss_fn = MultiTaskLossWrapper(classification=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    best_rmse = float("inf")
    best_weights = model.state_dict()

    save_path = os.path.join(
        save_path, str(datetime.now()).replace(":", "-").replace(" ", "_").split(".")[0]
    )
    writer = SummaryWriter(os.path.join(save_path, "tensorboard_log"))

    rmse_logs = []
    acc_logs = []
    f1_logs = []

    for i in range(n_epochs):
        start = time.time()
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
                    device=device,
                )

            else:
                model.eval()
                epoch_rmse, epoch_acc, epoch_f1 = validate_one_epoch(
                    epoch=i,
                    model=model,
                    dataloader=dataloader,
                    writer=writer,
                    loss_fn=loss_fn,
                    device=device,
                )
                rmse_logs.append(epoch_rmse)
                acc_logs.append(epoch_acc)
                f1_logs.append(epoch_f1)

                if epoch_rmse < best_rmse:
                    best_rmse = epoch_rmse
                    best_weights = model.state_dict()
                    save_model_weights(model.state_dict(), save_path, best=True)

                if epoch_rmse < target_rmse and epoch_acc > targer_acc and epoch_f1 > targer_f1:
                    save_model_weights(model.state_dict(), save_path, best=False)
                    print("Early Stopping at epoch", i, "\n")
                    return writer

        end = time.time()
        print(f"EPOCH[{i}] took {end - start:.2f} seconds", "\n")

    writer.close()
    df = pd.DataFrame(
        {"epoch": list(range(len(rmse_logs))), "rmse": rmse_logs, "acc": acc_logs, "f1": f1_logs}
    )
    df.to_csv(os.path.join(save_path, "log.csv"), index=False)
    save_model_weights(best_weights, save_path, best=False)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="훈련에 사용할 설정이 정의되어 있는 json file의 경로",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_mode", action="store_true")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        train_config = TrainConfig.from_json(json.load(f))

    if train_config.num_workers:
        torch.multiprocessing.set_start_method("spawn")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # model_config = load_model_config(train_config.model_name)

    if (
        train_config.cropper_weight_path
        and train_config.cropper_input_size
        and train_config.cropper_output_size
    ):
        dataloaders = make_dataloaders(
            image_meta_data_path=train_config.image_meta_data_path,
            img_dir=train_config.image_folder_path,
            num_classes=train_config.num_classes,
            on_memory=train_config.on_memory,
            test_size=train_config.test_size,
            batch_size=train_config.batch_size,
            num_workers=train_config.num_workers,
            train_transform=kim_aug(),
            cropper_weight_path=train_config.cropper_weight_path,
            cropper_input_size=train_config.cropper_input_size,
            cropper_output_size=train_config.cropper_output_size,
            test_mode=args.test_mode,
        )
    else:

        dataloaders = make_dataloaders(
            image_meta_data_path=train_config.image_meta_data_path,
            img_dir=train_config.image_folder_path,
            num_classes=train_config.num_classes,
            on_memory=train_config.on_memory,
            test_size=train_config.test_size,
            batch_size=train_config.batch_size,
            num_workers=train_config.num_workers,
            train_transform=kim_aug(),
            test_mode=args.test_mode,
        )

    model = make_model(
        model_name=train_config.model_name,
        num_classes=train_config.num_classes,
        hidden_size=1280,
    ).to(device)

    if train_config.weight_path:
        model.load_state_dict(torch.load(train_config.weight_path))

    train_and_valid(
        model=model,
        dataloaders=dataloaders,
        n_epochs=train_config.epoch,
        save_path=train_config.save_path,
        target_rmse=train_config.target_rmse,
        targer_acc=train_config.target_acc,
        targer_f1=train_config.target_f1,
        device=device,
    )
