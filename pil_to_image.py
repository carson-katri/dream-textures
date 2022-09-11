import bpy
import numpy as np

def pil_to_image(pil_image, name):
    """
    PIL image pixels is 2D array of byte tuple (when mode is 'RGB', 'RGBA') or byte (when mode is 'L')
    bpy image pixels is flat array of normalized values in RGBA order
    """
    from PIL import ImageOps
    width = pil_image.width
    height = pil_image.height
    byte_to_normalized = 1.0 / 255.0

    bpy_image = bpy.data.images.new(name, width=width, height=height)

    # Images are upside down for Blender, so use `ImageOps.flip` to fix it.
    bpy_image.pixels[:] = (np.asarray(ImageOps.flip(pil_image).convert('RGBA'),dtype=np.float32) * byte_to_normalized).ravel()
    bpy_image.pack()

    return bpy_image