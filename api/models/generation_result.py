from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
import math

@dataclass
class GenerationResult:
    """The output of a `Backend`.

    Create a result with an `image` and a `seed`.

    ```python
    result = GenerationResult(
        image=np.zeros((512, 512, 3)),
        seed=42
    )
    ```
    """
    image: NDArray
    """The generated image as a Numpy array.
    The shape should be `(height, width, channels)`, where `channels` is 3 or 4.
    """
    seed: int
    """The seed used to generate the image."""
    step: int
    """The step of generation"""

    @staticmethod
    def tile_images(results: list['GenerationResult']):
        images = [result.image for result in results]
        if len(images) == 0:
            return None
        elif len(images) == 1:
            return images[0]
        width = images[0].shape[1]
        height = images[0].shape[0]
        tiles_x = math.ceil(math.sqrt(len(images)))
        tiles_y = math.ceil(len(images) / tiles_x)
        tiles = np.zeros((height * tiles_y, width * tiles_x, 4), dtype=np.float32)
        bottom_offset = (tiles_x*tiles_y-len(images)) * width // 2
        for i, image in enumerate(images):
            x = i % tiles_x
            y = tiles_y - 1 - int((i - x) / tiles_x)
            x *= width
            y *= height
            if y == 0:
                x += bottom_offset
            tiles[y: y + height, x: x + width] = image
        return tiles