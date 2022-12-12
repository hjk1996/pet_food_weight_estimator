from torchvision import transforms
import torch.nn as nn


def kim_aug():
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

