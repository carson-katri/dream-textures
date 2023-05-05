from dataclasses import dataclass
from numpy.typing import NDArray

@dataclass
class GenerationResult:
    """The output of a `Backend`.

    Create a result with an `image` and a `seed`.

    ```python
    result = GenerationResult(
        progress=3,
        total=5,
        image=np.zeros((512, 512, 3)),
        seed=42
    )
    ```

    Alternatively, create a result with just a `title` and progress values.

    ```python
    result = GenerationResult(
        progress=3,
        total=5,
        title="Loading model"
    )
    ```
    """

    progress: int
    """The amount out of `total` that has been completed"""

    total: int
    """The number of steps to complete"""

    title: str | None = None
    """The name of the currently executing task"""
    
    image: NDArray | None = None
    """The generated image as a Numpy array.

    The shape should be `(height, width, channels)`, where `channels` is 3 or 4.
    """

    seed: int | None = None
    """The seed used to generate the image."""