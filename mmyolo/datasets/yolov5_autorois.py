# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets import XMLDataset

from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from ..registry import DATASETS


@DATASETS.register_module()
class AutoROISDataset(XMLDataset):
    """Dataset for PASCAL VOC."""

    METAINFO = {
        'CLASSES':
            ('main_driver', 'window', 'left_door', 'subdriver_sit'),
        # PALETTE is a list of color tuples, which is used for visualization.
        'PALETTE': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@DATASETS.register_module()
class YOLOv5AutoROISDataset(BatchShapePolicyDataset, AutoROISDataset):
    """Dataset for YOLOv5 VOC Dataset.

    We only add `BatchShapePolicy` function compared with VOCDataset. See
    `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    pass
