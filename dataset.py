import os
from typing import Tuple, List
from glob import glob


import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
from torchvision.transforms import Resize
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from models.yolo_wrapper import YOLOWrapper

class CustomDataset(Dataset):
    def __init__(
        self,
        meta_data: pd.DataFrame,
        img_dir: str,
        n_classes: int,
        device: torch.device,
        on_memory: bool = False,
        cropper: nn.Module = None,
        transform=None,
        resize: Tuple[int, int] = None,
    ):
        self.meta_data = meta_data
        self.img_dir = img_dir
        self.transform = transform
        self.n_classes = n_classes
        self.on_memory = on_memory
        self.cropper = cropper
        self.resizer = Resize(resize) if resize else None
        self.device = device
        
        if on_memory:
            self._load_image_on_memory()

    def _load_image_on_memory(self) -> None:
        self.img_tensors = []
        self.food_type_tensors = []
        self.gram_tensors = []

        for i in range(len(self.meta_data)):
            img_tensor = read_image(os.path.join(self.img_dir, self.meta_data.iloc[i, 2])).float() / 255
            self.img_tensors.append(img_tensor)

            food_type_tensor = torch.zeros(self.n_classes).type(torch.FloatTensor)
            food_type_tensor[list(map(int, str(self.meta_data.iloc[i, 1]).split()))] = 1.0   
            self.food_type_tensors.append(food_type_tensor)

            gram_tensor = torch.tensor([self.meta_data.iloc[i, 2]]).type(torch.FloatTensor)
            self.gram_tensors.append(gram_tensor)

            


    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        '''
        return: gram, food_type, image
        '''

        if self.on_memory:
            gram = torch.tensor(self.gram_tensors[idx])
            food_type = torch.tensor(self.food_type_tensors[idx])
            img = torch.tensor(self.img_tensors[idx])
            
        else:
            gram = torch.tensor([self.meta_data.iloc[idx, 2]]).type(torch.FloatTensor)
            food_type = torch.zeros(self.n_classes).type(torch.FloatTensor)
            food_type[list(map(int, str(self.meta_data.iloc[idx, 1]).split()))] = 1.0
            img_path = os.path.join(self.img_dir, self.meta_data.iloc[idx, 3])
            img = read_image(img_path).float() / 255


        
        if self.cropper:
            # cropper인 yolov7은 4차원의 입력을 받음 [batch_size, rgb, width, height]
            # 또한 입력값은 0과 1사이의 값으로 정규화된 float tensor임.
            # 따라서 cropping을 위해서 임시적으로 batch 차원을 추가해줘야함.
            img = self.cropper(img.unsqueeze(0))[0].squeeze()
                        
        if self.transform:
            # transform은 uint8(0~255) type의 텐서만 입력으로 받을 수 있음.
            img = torch.clamp((img * 255), min=0, max=255).type(torch.uint8)
            img = self.transform(img)

        if not self.cropper and self.resizer:
            img = self.resizer(img)

        img = img.float() / 255


        return gram.to(self.device), food_type.to(self.device), img.to(self.device)



def make_hash(df: pd.DataFrame) -> pd.Series:
    return (
        "bt"
        + df["bowl_type"].astype(str)
        + "ft"
        + df["food_type"].astype(str)
        + "gram"
        + df["gram"].astype(str)
    )


def make_dataloaders(
    meta_data_path: str,
    img_dir: str,
    num_classes: int,
    device: torch.device,
    test_size: float,
    batch_size: int,
    num_workers: int = 0,
    cropper_weight_path: str = None,
    cropper_input_size: int = None,
    cropper_output_size: int = None,
    on_memory: bool = False,
    transform=None,
    resize: Tuple[int, int] = None,
    random_state: int = 1234,
    test_mode: bool = False
) -> dict:
    meta_data = pd.read_csv(meta_data_path)
    hash = make_hash(meta_data)

    train, test = train_test_split(
        meta_data,
        test_size=test_size,
        random_state=random_state,
        stratify=hash,
    )

    if test_mode:
        train = train.iloc[: 256, :]
        test = test.iloc[: 32, :]
        
    cropper = YOLOWrapper(weight_path=cropper_weight_path, img_size=cropper_input_size, resize=cropper_output_size) if cropper_weight_path else None
    train_dataset = CustomDataset(
        train, img_dir, num_classes, device, transform=transform, cropper=cropper, resize=resize, on_memory=on_memory
    )
    test_dataset = CustomDataset(test, img_dir, num_classes, device, cropper=cropper,  resize=resize, on_memory=on_memory)
    train_dataloader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, num_workers=num_workers, batch_size=1, shuffle=True)

    return {"train": train_dataloader, "test": test_dataloader}
