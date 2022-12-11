import os

def absolute_path(component: str):
    """
    Returns the absolute path to a file in the addon directory.

    Alternative to `os.abspath` that works the same on macOS and Windows.
    """
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), component)

REAL_ESRGAN_WEIGHTS_PATH = absolute_path("weights/realesrgan/realesr-general-x4v3.pth")
CLIPSEG_WEIGHTS_PATH = absolute_path("weights/clipseg/rd64-uni.pth")