from dataclasses import dataclass
from numpy.typing import NDArray

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