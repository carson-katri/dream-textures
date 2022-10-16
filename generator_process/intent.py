from enum import IntEnum
from math import ceil, log

class Intent(IntEnum):
    """IPC messages types sent from frontend to backend"""
    UNKNOWN = -1

    PROMPT_TO_IMAGE = 0
    UPSCALE = 1
    STOP = 2
    APPLY_OCIO_TRANSFORMS = 3

    @classmethod
    def _missing_(cls, value):
        return cls.UNKNOWN
INTENT_BYTE_LENGTH = ceil(log(max(Intent)+1,256))