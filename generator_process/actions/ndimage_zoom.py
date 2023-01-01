import numpy as np
from numpy.typing import NDArray
from typing import Tuple

def ndimage_zoom(
    self,
    image: NDArray,
    size: Tuple[int, int]
) -> NDArray:
    from scipy.ndimage import zoom
    return zoom(image, np.divide(size, image.shape))