from enum import Enum
from typing import List, Optional, Tuple, Dict
from torchvision import transforms
import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F



def kim_aug():
    random_apply1 = transforms.RandomApply(nn.ModuleList([
                            transforms.ColorJitter(brightness=0.5)]), 
                            p=0.5)
    random_apply2 = transforms.RandomApply(nn.ModuleList([
                            transforms.RandomRotation(degrees=(-45, 45))]),
                            p=0.5)          

    return transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            random_apply1,
                            random_apply2,
                            ])

