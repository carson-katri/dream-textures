from enum import IntEnum
from math import ceil, log

class Action(IntEnum):
    """IPC message types sent from backend to frontend"""
    UNKNOWN = -1 # placeholder so you can do Action(int).name or Action(int) == Action.UNKNOWN when int is invalid
                 # don't add anymore negative actions
    CLOSED = 0 # is not sent during normal operation, just allows for a simple way of detecting when the subprocess is closed
    INFO = 1
    IMAGE = 2
    STEP_IMAGE = 3
    STEP_NO_SHOW = 4
    EXCEPTION = 5
    STOPPED = 6

    @classmethod
    def _missing_(cls, value):
        return cls.UNKNOWN
ACTION_BYTE_LENGTH = ceil(log(max(Action)+1,256))