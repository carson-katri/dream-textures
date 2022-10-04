import os

def absolute_path(component):
    """
    Returns the absolute path to a file in the addon directory.

    Alternative to `os.abspath` that works the same on macOS and Windows.
    """
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), component)

WEIGHTS_PATH = absolute_path("stable_diffusion/models/ldm/stable-diffusion-v1/model.ckpt")
REAL_ESRGAN_WEIGHTS_PATH = absolute_path("weights/realesrgan/realesr-general-x4v3.pth")