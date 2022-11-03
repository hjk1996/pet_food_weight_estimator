import os

import pandas as pd
import torch
import torch
from torchvision.transforms import Resize
from torchvision import transforms, models
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(
        self,
        meta_data: pd.DataFrame,
        img_dir: str,
        n_classes: int,
        device: torch.device,
        transform=None,
    ):
        self.meta_data = meta_data
        self.img_dir = img_dir
        self.transform = transform
        self.n_classes = n_classes
        self.device = device

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        food_type = torch.zeros(self.n_classes).type(torch.FloatTensor)
        # a = self.meta_data.iloc[idx, 0]
        # print(a)
        # print(type(a))
        food_type[self.meta_data.iloc[idx, 0]] = 1.0

        weight = torch.tensor([self.meta_data.iloc[idx, 1]]).type(torch.FloatTensor)

        img_path = os.path.join(self.img_dir, self.meta_data.iloc[idx, 2])
        img = read_image(img_path)

        if self.transform:
            img = self.transform(img)

        img = img.type(torch.FloatTensor) / 255.0

        return weight.to(self.device), food_type.to(self.device), img.to(self.device)


def load_meta_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)[["food_type", "gram", "image_name", "hash"]]


def make_dataloaders(
    meta_data_path: str,
    img_dir: str,
    n_classes: int,
    device: torch.device,
    test_size: float,
    batch_size: int,
    transform=None,
    random_state: int = 1234,
) -> dict:
    meta_data = load_meta_data(meta_data_path)
    train, test = train_test_split(meta_data, test_size=test_size, random_state=random_state)
    train = train[:100]
    test = test[:100]
    train_dataset = CustomDataset(
        train, img_dir, n_classes, device, transform=transform
    )
    test_dataset = CustomDataset(test, img_dir, n_classes, device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return {"train": train_dataloader, "test": test_dataloader}
