import os
from typing import Tuple, List
from glob import glob

from PIL import Image
import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
from torchvision.transforms import Resize, ToPILImage, ToTensor
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from models.yolo_wrapper import YOLOWrapper



# read image using pil
def read_image_as_pil(img_path: str) -> Image:
    return Image.open(img_path).convert("RGB")



class YOLODataset(Dataset):
    


    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        device: torch.device,
        on_memory: bool = False,
    ):
        self.img_dir = glob(os.path.join(img_dir, '*.jpg'))
        self.label_dir = label_dir
        self.get_coords()
        self.on_memory = on_memory
        self.device = device

    def __len__(self):
        return len(self.img_dir)
    
    def __getitem__(self, i):
        '''
            return
                
                img_tensor: [3, width, height]
                coords_tensor: [xyxy]
        '''
        img_name = self.img_dir[i]
        img_tensor = read_image(img_name).float() / 255
        coords_tensor = self.coords_map[os.path.basename(self.img_dir[i]).split('.')[0]]
        return img_name, img_tensor, coords_tensor

    def get_coords(self):
        self.coords_map = {}
        labels_file_paths = glob(os.path.join(self.label_dir, "*.txt"))
        for path in labels_file_paths:
            with open(path, 'r') as f:
                self.coords_map[os.path.basename(path).split('.')[0]] = torch.tensor(list(map(float, f.readline().split()[1:])))

class DogFoodDataset(Dataset):
    def __init__(
        self,
        meta_data: pd.DataFrame,
        img_dir: str,
        n_classes: int,
        on_memory: bool = False,
        cropper: nn.Module = None,
        transform=None,
    ):
        self.meta_data = meta_data
        self.img_dir = img_dir
        self.transform = transform
        self.n_classes = n_classes
        self.on_memory = on_memory
        self.cropper = cropper
        
        if on_memory:
            self._load_image_on_memory()

    def _load_image_on_memory(self) -> None:
        self.img_tensors = []
        self.food_type_tensors = []
        self.gram_tensors = []

        for i in range(len(self.meta_data)):
            img_tensor = read_image(os.path.join(self.img_dir, self.meta_data.iloc[i, 3])).float() / 255
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
            gram = self.gram_tensors[idx].clone().detach()
            food_type = self.food_type_tensors[idx].clone().detach()
            img = self.img_tensors[idx].clone().detach()
            
        else:
            gram = torch.tensor([self.meta_data.iloc[idx, 2]]).type(torch.FloatTensor)
            food_type = torch.zeros(self.n_classes).type(torch.FloatTensor)
            food_type[list(map(int, str(self.meta_data.iloc[idx, 1]).split()))] = 1.0
            img_path = os.path.join(self.img_dir, self.meta_data.iloc[idx, 3])
            img = read_image(img_path) 


        
        if self.cropper:
            # cropper인 yolov7은 4차원의 입력을 받음 [batch_size, rgb, width, height]
            # 또한 입력값은 0과 1사이의 값으로 정규화된 float tensor임.
            # 따라서 cropping을 위해서 임시적으로 batch 차원을 추가해줘야함.
            img = img.float() / 255
            img = self.cropper(img.unsqueeze(0))[0].squeeze()
            img = torch.clamp((img * 255), min=0, max=255).type(torch.uint8)
        
        if self.transform:
            # transform은 uint8(0~255) type의 텐서만 입력으로 받을 수 있음.
            img = self.transform(img)

        img = img.float() / 255

        return gram, food_type, img



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
    image_meta_data_path: str,
    img_dir: str,
    num_classes: int,
    test_size: float,
    batch_size: int,
    num_workers: int = 0,
    cropper_weight_path: str = None,
    cropper_input_size: int = None,
    cropper_output_size: int = None,
    on_memory: bool = False,
    train_transform=None,
    val_transform=None,
    random_state: int = 1234,
    test_mode: bool = False
) -> dict:
    meta_data = pd.read_csv(image_meta_data_path)
    hash = make_hash(meta_data)

    train, test = train_test_split(
        meta_data,
        test_size=test_size,
        random_state=random_state,
        stratify=hash,
    )

    if test_mode:
        train = train.iloc[:128, :]
        test = test.iloc[:64, :]
        
    cropper = YOLOWrapper(weight_path=cropper_weight_path, img_size=cropper_input_size, resize=cropper_output_size) if cropper_weight_path else None
    train_dataset = DogFoodDataset(
        train, img_dir, num_classes, transform=train_transform, cropper=cropper,  on_memory=on_memory
    )
    test_dataset = DogFoodDataset(test, img_dir, num_classes, transform=val_transform, cropper=cropper,  on_memory=on_memory)
    train_dataloader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)

    return {"train": train_dataloader, "test": test_dataloader}

def make_dataloaders_for_cv10(
    image_meta_data_path: str,
    img_dir: str,
    num_classes: int,
    batch_size: int,
    num_workers: int = 0,
    cropper_weight_path: str = None,
    cropper_input_size: int = None,
    cropper_output_size: int = None,
    on_memory: bool = False,
    transform=None,
    test_mode: bool = False
) -> List[dict]:
    dataset_list = []
    meta_data = pd.read_csv(image_meta_data_path)
    cropper = YOLOWrapper(weight_path=cropper_weight_path, img_size=cropper_input_size, resize=cropper_output_size) if cropper_weight_path else None
    skf = StratifiedKFold(n_splits=10, )

    for train_index, test_index in skf.split(meta_data,  meta_data['gram'].map(str) + meta_data['food_type']):
        train = meta_data.iloc[train_index, :]
        test = meta_data.iloc[test_index, :]
        
        if test_mode:
            train = train[:128]
            test = test[:64]

        train_dataset = DogFoodDataset(
            train, img_dir, num_classes,  transform=transform, cropper=cropper,  on_memory=on_memory
        )
        test_dataset = DogFoodDataset(test, img_dir, num_classes,  cropper=cropper,  on_memory=on_memory)
        train_dataloader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
        dataset_list.append({"train": train_dataloader, "test": test_dataloader})
    
    return dataset_list
