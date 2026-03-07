# Lazy imports to avoid pulling in heavy dependencies at package load time.


def __getattr__(name):
    if name == "BearSegmentationDataset":
        from .dataset import BearSegmentationDataset
        return BearSegmentationDataset
    if name == "SegmentationTrainer":
        from .train_segmentation import SegmentationTrainer
        return SegmentationTrainer
    if name == "BearSegmentor":
        from .infer_segmentation import BearSegmentor
        return BearSegmentor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
