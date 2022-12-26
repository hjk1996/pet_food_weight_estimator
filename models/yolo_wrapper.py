from typing import Tuple, Optional
import torch
import torch.nn as nn
import time
import numpy as np
from yolov7.yolo_models.experimental import attempt_load
from torchvision.transforms.transforms import Resize
import torchvision
import torchvision.transforms.functional as T


class YOLOWrapper(nn.Module):
    def __init__(
        self, weight_path: str, img_size: int, resize: Optional[int] = None
    ) -> None:
        super().__init__()
        self.model = attempt_load(weight_path)
        self.img_size = img_size
        self.resize = resize
        self.resizer = Resize((img_size, img_size)) if resize is not None else None
        self.crop = T.crop

    def forward(self, x):
        """
        return: 
            images [batch, 3, w, h]

            has_bowl [batch, 1]

        """

        with torch.no_grad():
            preds, _ = self.model(x)
            preds = self.non_max_suppression(preds, max_det=1)
            images = torch.zeros((len(preds), 3, self.resize, self.resize))
            has_bowl = torch.zeros((len(preds), 1))

            for i, pred in enumerate(preds):
                if pred.shape[0] == 0:
                    images[i] = self.resizer(x[i])
                    has_bowl[i, 0] = False
                else:
                    self.clip_coords(pred[:, :4], (self.img_size, self.img_size))
                    x1, y1, x2, y2 = pred[:, :4].round().int().view(-1).tolist()
                    w = x2 - x1
                    h = y2 - y1
                    cropped = self.crop(x[i], y1, x1, h, w)
                    if self.resize:
                        resized = self.resizer(cropped)
                    else:
                        resized = cropped
                    images[i] = resized
                    has_bowl[i, 0] = True

        return images, has_bowl

    def get_miou(
        self, image_tensor: torch.tensor, box_coords: torch.tensor
    ) -> Tuple[list, float]:
        """
        input:

            image_tensor: [batch_size, 3, width, hegiht]

            box_coords: [batch_size, xywh]

            하나의 이미지에는 하나의 좌표값만 부여됨.

            좌표값이 없는 이미지는 없음 (이미지에 개밥그릇 사진이 무조건 있어야함.)
        

        output:

            mIoU: float [0,1]
        """
        assert image_tensor.shape[0] == box_coords.shape[0]

        with torch.no_grad():
            preds, _ = self.model(image_tensor)
            # preds는 각 배치에 대한 예측 결과를 담은 리스트임
            # 즉, preds의 length는 배치 사이즈와 같음.
            preds = self.non_max_suppression(preds, max_det=1)
            ious = []
            pred_coords = []

            for i, pred in enumerate(preds):
                if pred.shape[0] == 0:
                    ious.append(0)
                    pred_coords.append([0, 0, 0, 0])
                else:
                    self.clip_coords(pred[:, :4], (self.img_size, self.img_size))
                    real_box_coords = (
                        self.xywh2xyxy(box_coords[i].unsqueeze(0)) * self.img_size
                    )
                    iou = self.box_iou(pred[:, :4], real_box_coords).squeeze().item()
                    ious.append(iou)
                    pred_coords.append(pred[:, :4].squeeze().tolist())

        return ious, sum(ious) / image_tensor.shape[0], pred_coords

    def non_max_suppression(
        self,
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        max_det=300,
        labels=(),
    ):

        """Runs Non-Maximum Suppression (NMS) on inference results

        Returns:
            shape: [batch_size, n_classes, 6]
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        # [x, y, x, y, confidence, ...classes]
        nc = prediction.shape[2] - 5  # number of classes
        # confidence가 기준치 이상인 것들에 대한 indices
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_det = max_det  # maximum number of detections per image
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            # confidence 기준치 이상인 것들만으로 거름
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 5), device=x.device)
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            if nc == 1:
                x[:, 5:] = x[
                    :, 4:5
                ]  # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                # so there is no need to multiplicate.
            else:
                x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                    1, keepdim=True
                )  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                print(f"WARNING: NMS time limit {time_limit}s exceeded")
                break  # time limit exceeded

        return output

    @staticmethod
    def box_iou(box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        def box_area(box):
            # box: (x, y, x, y)
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (
            (
                torch.min(box1[:, None, 2:], box2[:, 2:])
                - torch.max(box1[:, None, :2], box2[:, :2])
            )
            .clamp(0)
            .prod(2)
        )
        return inter / (
            area1[:, None] + area2 - inter
        )  # iou = inter / (area1 + area2 - inter)

    @staticmethod
    def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(
                img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
            )  # gain  = old / new
            pad = (
                (img1_shape[1] - img0_shape[1] * gain) / 2,
                (img1_shape[0] - img0_shape[0] * gain) / 2,
            )  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords

    @staticmethod
    def clip_coords(boxes, img_shape):
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2
