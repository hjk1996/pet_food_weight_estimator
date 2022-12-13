from enum import Enum
from typing import List, Optional, Tuple, Dict
from torchvision import transforms
import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F



def kim_aug():
    transforms.AugMix
    random_apply1 = transforms.RandomApply(nn.ModuleList([
                            transforms.ColorJitter(brightness=0.2)]), 
                            p=0.3)
    random_apply2 = transforms.RandomApply(nn.ModuleList([
                            transforms.RandomRotation(degrees=(-30, 30))]),
                            p=0.3)          

    return transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            random_apply1,
                            random_apply2,
                            ])

