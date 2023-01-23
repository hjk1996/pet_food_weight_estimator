import os
import torch
from models.yolo_wrapper import YOLOWrapper
from dataset import ImageDataset
import argparse
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight_path", type=str, required=True, help="path to weight file"
    )
    parser.add_argument("--img_size", type=int, default=416, help="image size")
    parser.add_argument(
        "--img_dir", type=str, required=True, help="path to image directory"
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="path to save txt files"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOWrapper(weight_path=args.weight_path, img_size=args.img_size).to(device)

    dataset = ImageDataset(
        img_dir=args.img_dir,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
    )

    coords_dict = {}

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    for img_name, img_tensor in tqdm(dataloader, desc="Generating coords"):
        img_tensor = img_tensor.to(device)
        base_names = [os.path.basename(name) for name in img_name]
        coords = model.get_xywh_coords(base_names, img_tensor)
        coords_dict.update(coords)

    for img_name, coords in tqdm(coords_dict.items(), desc="Saving coords"):
        with open(os.path.join(args.save_path, img_name + ".txt"), "w") as f:
            f.write("0 " + " ".join(map(str, coords)))
