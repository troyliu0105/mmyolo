# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets import XMLDataset

from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from ..registry import DATASETS


@DATASETS.register_module()
class ChatPhoneNGDataset(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'CLASSES':
            ('phone', 'head_left', 'head_straight', 'head_right'),
        # PALETTE is a list of color tuples, which is used for visualization.
        'PALETTE': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@DATASETS.register_module()
class YOLOv5ChatPhoneNGDataset(BatchShapePolicyDataset, ChatPhoneNGDataset):
    """Dataset for YOLOv5 VOC Dataset.

    We only add `BatchShapePolicy` function compared with VOCDataset. See
    `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    pass