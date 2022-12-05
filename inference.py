import json
from typing import Union

import argparse
import torch

from utils import (
    load_image_as_tensor,
    make_swin_v2_based_estimator,
    get_class_prediction_from_logit,
    load_model_config,
    InferenceConfig,
    ModelConfig

)


def inference_on_one_image(
    model_config: ModelConfig,
    image_path: str,
    weight_path: str,
    num_classes: int,
    mapping_path: str = None,
    resize: int = None,
) -> Union[tuple, int]:
    print(f"{image_path}에 대한 예측 시작..\n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = load_image_as_tensor(image_path, resize=resize)
    model = make_swin_v2_based_estimator(device=device, model_config=model_config,num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weight_path))

    weight, class_logit = model(img)
    weight = round(weight)
    class_pred = get_class_prediction_from_logit(class_logit)

    if mapping_path:
        with open(mapping_path, "r") as f:
            indice_to_name = json.load(f)
            class_pred = [indice_to_name[str(i)] for i in class_pred]

    print(f"무게: {weight} gram\n")
    print(f"사료 종류: {class_pred}")
    return weight, class_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config_path", type=str, default='./inference_config.json', help="예측 설정 json 파일의 경로")
    parser.add_argument("--image_path", type=str, required=True, help="예측할 이미지의 경로")
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="값을 입력하면 해당 사이즈로 이미지를 리사이즈한 후 모델의 입력으로 전달함",
    )

    args = parser.parse_args()

    
    with open(args.inference_config_path, 'r') as f:
        infer_config = InferenceConfig.from_json(json.load(f))
        model_config = load_model_config(infer_config.model_name)


    results = inference_on_one_image(
        model_config=model_config,
        image_path=args.image_path,
        weight_path=infer_config.weight_path,
        num_classes=infer_config.num_classes,
        mapping_path=infer_config.mapping_path,
        resize=args.resize,
    )
