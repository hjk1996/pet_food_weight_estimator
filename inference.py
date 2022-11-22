import json
from typing import Union

import argparse
import torch
from utils import (
    load_image_as_input_tensor,
    make_swin_v2_based_estimator,
    get_class_prediction_from_logit,
)


def inference_on_one_image(
    image_path: str,
    weights_path: str,
    n_classes: int,
    mapping_path: str = None,
    resize: int = None,
) -> Union[tuple, int]:
    print(f"{image_path}에 대한 예측 시작..\n")
    device = torch.device("cpu")
    img = load_image_as_input_tensor(image_path, resize=resize)
    model = make_swin_v2_based_estimator(device=device, n_classes=n_classes).to(device)
    model.load_state_dict(torch.load(weights_path))

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
    parser.add_argument("--image_path", type=str, required=True, help="예측할 이미지의 경로")
    parser.add_argument("--weights", type=str, required=True, help="예측에 사용할 모델 가중치의 경로")
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="값을 입력하면 해당 사이즈로 이미지를 리사이즈한 후 모델의 입력으로 전달함",
    )
    parser.add_argument("--n_classes", type=int, default=21, help="사료 종류 수")
    parser.add_argument(
        "--mapping_path", type=str, default=None, help="사료의 인덱스와 이름을 매핑한 json 파일의 경로"
    )
    args = parser.parse_args()

    results = inference_on_one_image(
        image_path=args.image_path,
        weights_path=args.weights,
        classification=args.classification,
        n_classes=args.n_classes,
        mapping_path=args.mapping_path,
        resize=args.resize,
    )
