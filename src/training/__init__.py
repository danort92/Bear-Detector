# Lazy imports to avoid pulling in heavy dependencies (torch, ultralytics)
# at package load time in environments where they may not be installed.


def __getattr__(name):
    if name == "ClassificationTrainer":
        from .train_classification import ClassificationTrainer
        return ClassificationTrainer
    if name == "DetectionTrainer":
        from .train_detection import DetectionTrainer
        return DetectionTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
