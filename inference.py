import json
from typing import Union
import time
import os

import argparse
import torch

from utils import (
    load_image_as_tensor,
    get_class_prediction_from_logit,
    InferenceConfig,
)

from models.model_loader import make_model


def inference_on_one_image(
    model_name: str,
    image_path: str,
    num_classes: int,
    weight_path: str = None,
    mapping_path: str = None,
    resize: int = None,
) -> Union[tuple, int]:
    print(f"{os.path.basename(image_path)}에 대한 예측 시작..\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = load_image_as_tensor(image_path, resize=resize).to(device)
    model = make_model(model_name=model_name, num_classes=num_classes).to(device)

    if weight_path:
        model.load_state_dict(torch.load(weight_path))

    indice_to_name = None
    if mapping_path:
        with open(mapping_path, "r") as f:
            indice_to_name = json.load(f)

    model.eval()

    # warm up
    sample = torch.zeros_like(img).to(device)
    _, _ = model(sample)

    start = time.time()
    weight, class_logit = model(img)
    weight = round(weight.item())
    class_pred = get_class_prediction_from_logit(class_logit)[0]
    if indice_to_name:
        class_pred = [indice_to_name[str(i)] for i in class_pred]
    end = time.time()
    print(f"무게: {weight} gram\n")
    print(f"사료 종류: {class_pred}")
    print(f"예측에 걸린 시간: {(end - start) / 1000:4f} (ms)\n")

    return weight, class_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="./inference_config.json",
        help="예측 설정 json 파일의 경로",
    )
    parser.add_argument("--img_path", type=str, required=True, help="예측할 이미지의 경로")
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="값을 입력하면 해당 사이즈로 이미지를 리사이즈한 후 모델의 입력으로 전달함",
    )

    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        infer_config = InferenceConfig.from_json(json.load(f))

    results = inference_on_one_image(
        model_name=infer_config.model_name,
        image_path=args.img_path,
        weight_path=infer_config.weight_path,
        num_classes=infer_config.num_classes,
        mapping_path=infer_config.mapping_path,
        resize=args.resize,
    )
