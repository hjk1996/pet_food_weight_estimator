import json
import argparse

import torch
import torchvision.transforms as T

from utils import load_model_config, TrainConfig, make_swin_v2_based_estimator
from dataset import make_dataloaders_for_cv10
from train import train_and_valid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config_path", type=str, required=True,  help="훈련에 사용할 설정이 정의되어 있는 json file의 경로")
    args = parser.parse_args()

    with open(args.train_config_path, "r") as f:
        train_config = TrainConfig.from_json(json.load(f))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_config = load_model_config(train_config.model_name)

    if train_config.cropper_weight_path and train_config.cropper_input_size and train_config.cropper_output_size:
        
        dataset_list = make_dataloaders_for_cv10(
            image_meta_data_path=train_config.image_meta_data_path,
            img_dir=train_config.image_folder_path,
            num_classes=train_config.num_classes,
            device=device,
            on_memory=train_config.on_memory,
            batch_size=train_config.batch_size,
            num_workers=train_config.num_classes,
            transform=T.AugMix(),
            cropper_weight_path=train_config.cropper_weight_path,
            cropper_input_size=train_config.cropper_input_size,
            cropper_output_size=train_config.cropper_output_size
        )
    else:
        dataset_list = make_dataloaders_for_cv10(
            image_meta_data_path=train_config.image_meta_data_path,
            img_dir=train_config.image_folder_path,
            num_classes=train_config.num_classes,
            device=device,
            on_memory=train_config.on_memory,
            batch_size=train_config.batch_size,
            num_workers=train_config.num_classes,
            transform=T.AugMix(),
        )
    

    for dataloaders in dataset_list:
        model = make_swin_v2_based_estimator(device=device, model_config=model_config, num_classes=train_config.num_classes)

        train_and_valid(
            model=model,
            dataloaders=dataloaders,
            n_epochs=train_config.epoch,
            save_path=train_config.save_path
        )