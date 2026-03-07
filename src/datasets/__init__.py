# Lazy imports so torch/torchvision are only pulled in when actually used.


def __getattr__(name):
    if name in ("BearClassificationDataset", "build_classification_dataloaders"):
        from .classification_dataset import (
            BearClassificationDataset,
            build_classification_dataloaders,
        )
        return locals()[name]
    if name == "BearDetectionDataset":
        from .detection_dataset import BearDetectionDataset
        return BearDetectionDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
