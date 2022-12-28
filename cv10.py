import json
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from utils import TrainConfig
from dataset import make_dataloaders_for_cv10
from train import train_and_valid
from augmentation import kim_aug
from models.model_loader import make_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="훈련에 사용할 설정이 정의되어 있는 json file의 경로",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
    )
    parser.add_argument("--test_mode", action="store_true")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        train_config = TrainConfig.from_json(json.load(f))

    if train_config.num_workers:
        torch.multiprocessing.set_start_method("spawn")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if (
        train_config.cropper_weight_path
        and train_config.cropper_input_size
        and train_config.cropper_output_size
    ):

        dataset_list = make_dataloaders_for_cv10(
            image_meta_data_path=train_config.image_meta_data_path,
            img_dir=train_config.image_folder_path,
            num_classes=train_config.num_classes,
            on_memory=train_config.on_memory,
            batch_size=train_config.batch_size,
            num_workers=train_config.num_workers,
            transform=kim_aug(),
            cropper_weight_path=train_config.cropper_weight_path,
            cropper_input_size=train_config.cropper_input_size,
            cropper_output_size=train_config.cropper_output_size,
            test_mode=args.test_mode,
        )
    else:
        dataset_list = make_dataloaders_for_cv10(
            image_meta_data_path=train_config.image_meta_data_path,
            img_dir=train_config.image_folder_path,
            num_classes=train_config.num_classes,
            on_memory=train_config.on_memory,
            batch_size=train_config.batch_size,
            num_workers=train_config.num_workers,
            transform=kim_aug(),
            test_mode=args.test_mode,
        )

    dfs = []
    for dataloaders in dataset_list:
        model = make_model(
            model_name=train_config.model_name, num_classes=train_config.num_classes
        ).to(device)

        df = train_and_valid(
            model=model,
            dataloaders=dataloaders,
            n_epochs=train_config.epoch,
            save_path=train_config.save_path,
            device=device,
            target_rmse=train_config.target_rmse,
            targer_acc=train_config.target_acc,
            targer_f1=train_config.target_f1,
        )
        dfs.append(df)

    lowest_rmse = list(map(lambda x: x["rmse"].min(), dfs))
    highest_acc = list(map(lambda x: x["acc"].max(), dfs))
    highest_f1 = list(map(lambda x: x["f1"].max(), dfs))

    cv10_log_folder_path = os.path.join(train_config.save_path, "cv10_log")
    os.makedirs(cv10_log_folder_path, exist_ok=True)

    for i, df in enumerate(dfs):
        df.to_csv(os.path.join(cv10_log_folder_path, f"fold_{i+1}.csv"), index=False)

    df = pd.DataFrame(
        {
            "fold": [i + 1 for i in range(len(lowest_rmse))],
            "lowest_rmse": lowest_rmse,
            "highest_acc": highest_acc,
            "highest_f1": highest_f1,
            "mean_rmse": [np.mean(lowest_rmse)] * len(lowest_rmse),
            "mean_acc": [np.mean(highest_acc)] * len(highest_acc),
            "mean_f1": [np.mean(highest_f1)] * len(highest_f1),
        }
    )

    df.to_csv(os.path.join(cv10_log_folder_path, "cv10_summary.csv"), index=False)
