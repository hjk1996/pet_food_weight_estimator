import os
import argparse
from datetime import datetime
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.yolo_wrapper import YOLOWrapper
from dataset import YOLODataset

# make timestamp format as string date that can be used as file name
def timestamp_to_file_name(timestamp) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path", type=str, default="./yolov7/weights/iitp/best.pt")
    parser.add_argument(
        "--img_path", type=str, default="./data/iitp/yolo_data/test/images"
    )
    parser.add_argument(
        "--label_path", type=str, default="./data/iitp/yolo_data/test/labels"
    )
    parser.add_argument("--save_path", type=str, default="./results/yolov7")
    parser.add_argument("--img_size", type=int, default=416)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo = YOLOWrapper(weight_path=args.weight_path, img_size=args.img_size,).to(device)

    dataset = YOLODataset(img_dir=args.img_path, label_dir=args.label_path)

    dataloader = DataLoader(dataset=dataset, batch_size=8,)
    names = []
    ious = []
    real_x1 = []
    real_y1 = []
    real_x2 = []
    real_y2 = []
    pred_x1 = []
    pred_y1 = []
    pred_x2 = []
    pred_y2 = []
    for name, img, coords in tqdm(dataloader, desc="Calculating mIOU"):
        img, coords = img.to(device), coords.to(device)
        batch_ious, miou, batch_coords = yolo.get_miou(img, coords)
        names += name
        ious += batch_ious
        real_x1 += coords[:, 0].tolist()
        real_y1 += coords[:, 1].tolist()
        real_x2 += coords[:, 2].tolist()
        real_y2 += coords[:, 3].tolist()
        pred_x1 += list(map(lambda x: x[0] / args.img_size, batch_coords))
        pred_y1 += list(map(lambda x: x[1] / args.img_size, batch_coords))
        pred_x2 += list(map(lambda x: x[2] / args.img_size, batch_coords))
        pred_y2 += list(map(lambda x: x[3] / args.img_size, batch_coords))
        print(f"current miou: {sum(ious) / len(ious)}")

    log = pd.DataFrame(
        {
            "name": names,
            "iou": ious,
            "real_x1": real_x1,
            "real_y1": real_y1,
            "real_x2": real_x2,
            "real_y2": real_y2,
            "pred_x1": pred_x1,
            "pred_y1": pred_y1,
            "pred_x2": pred_x2,
            "pred_y2": pred_y2,
        }
    )

    print(f"miou: {log['iou'].mean()}")
    save_path = os.path.join(
        args.save_path, f"{timestamp_to_file_name(datetime.now().timestamp())}.csv"
    )
    log.to_csv(save_path, index=False)
    print(f"saved at {save_path}")
