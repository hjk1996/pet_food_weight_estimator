import imp
import torch

# from models.experimental import attempt_load
import torch
from torchvision.io import read_image
from torchvision.transforms import Resize
from datetime import datetime

# from yolo_utils.general import non_max_suppression, scale_coords

if __name__ == "__main__":
    #  device = torch.device("cpu")
    #  detector = attempt_load("./model_weights/best.pt", map_location=device)
    #  resizer = Resize((416, 416))
    #  sample_image = read_image(".\\data\\images\\20220930_172107.jpg")
    #  sample_image = resizer(sample_image).float().unsqueeze(0)
    #  sample_image /= 0.255
    #  with torch.no_grad():
    #      pred = detector(sample_image, augment=False)[0]
    #  # print(result.size())
    #  pred = non_max_suppression(pred)

    #  for det in pred:
    #      if len(det):
    #          det[:, :4] = scale_coords(sample_image.shape[2:], det[:, :4], (416, 416))

    #          for *xyxy, conf, cls in reversed(det):
    #              pass

    a = str(datetime.now()).replace(":", "-").replace(" ", "_").split(".")[0]
    print(a)
