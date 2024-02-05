import copy
from .yolov5_backbone import YoloV5BackBone
from .resnet import resnet50

def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    # name = backbone_cfg.pop("name")
    name = backbone_cfg.Model.Backbone.name
    if name == "YoloV5":
        return YoloV5BackBone(backbone_cfg)
    elif name == "ResNet50":
        return resnet50(backbone_cfg)
    else:
        raise NotImplementedError