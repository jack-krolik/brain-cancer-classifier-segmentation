from enum import StrEnum, auto

class TaskType(StrEnum):
    TUMOR_CLASSIFICATION = auto()
    TUMOR_SEGMENTATION = auto()

class DataSplit(StrEnum):
    TRAIN = auto()
    TEST = auto()
    VALIDATION = auto()
