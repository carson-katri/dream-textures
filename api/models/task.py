from dataclasses import dataclass
from typing import Tuple
from numpy.typing import NDArray
from enum import IntEnum

class Task:
    """A specific task type.
    
    Access the properties of the task using dot notation.
    
    ```python
    # Task.ImageToImage
    task.image
    task.strength
    task.fit
    ```

    Switch over the task to perform the correct actions.

    ```python
    match type(task):
        case PromptToImage:
            ...
        case ImageToImage:
            ...
        case Inpaint:
            ...
        case DepthToImage:
            ...
        case Outpaint:
            ...
    ```
    """

    @classmethod
    def name(cls) -> str:
        "unknown"
    """A human readable name for this task."""

@dataclass
class PromptToImage(Task):
    @classmethod
    def name(cls):
        return "prompt to image"

@dataclass
class ImageToImage(Task):
    image: NDArray
    strength: float
    fit: bool

    @classmethod
    def name(cls):
        return "image to image"

@dataclass
class Inpaint(ImageToImage):
    class MaskSource(IntEnum):
        ALPHA = 0
        PROMPT = 1

    mask_source: MaskSource
    mask_prompt: str
    confidence: float

    @classmethod
    def name(cls):
        return "inpainting"

@dataclass
class DepthToImage(Task):
    depth: NDArray | None
    image: NDArray | None
    strength: float

    @classmethod
    def name(cls):
        return "depth to image"

@dataclass
class Outpaint(Task):
    image: NDArray
    origin: Tuple[int, int]

    @classmethod
    def name(cls):
        return "outpainting"

@dataclass
class Upscale(Task):
    image: NDArray
    tile_size: int
    blend: int

    @classmethod
    def name(cls):
        return "upscaling"