import enum

class StepPreviewMode(enum.Enum):
    NONE = "None"
    FAST = "Fast"
    FAST_BATCH = "Fast (Batch Tiled)"
    ACCURATE = "Accurate"
    ACCURATE_BATCH = "Accurate (Batch Tiled)"