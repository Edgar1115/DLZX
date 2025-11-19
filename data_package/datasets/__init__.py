# data_package/datasets/__init__.py

from .classification import ImageClassificationDataset
from .segmentation import ImageSegmentationDataset

__all__ = [
    "ImageClassificationDataset",
    "ImageSegmentationDataset",
]
