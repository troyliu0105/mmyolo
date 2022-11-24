# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets import XMLDataset

from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from ..registry import DATASETS


@DATASETS.register_module()
class ChatPhoneDataset(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'CLASSES':
            ('front', 'profile', 'phone'),
        # PALETTE is a list of color tuples, which is used for visualization.
        'PALETTE': [(106, 0, 228), (119, 11, 32), (165, 42, 42)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@DATASETS.register_module()
class YOLOv5ChatPhoneDataset(BatchShapePolicyDataset, ChatPhoneDataset):
    """Dataset for YOLOv5 VOC Dataset.

    We only add `BatchShapePolicy` function compared with VOCDataset. See
    `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    pass
