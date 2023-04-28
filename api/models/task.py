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
    pass

@dataclass
class PromptToImage(Task):
    pass

@dataclass
class ImageToImage(Task):
    image: NDArray
    strength: float
    fit: bool

@dataclass
class Inpaint(Task):
    class MaskSource(IntEnum):
        ALPHA = 0
        PROMPT = 1

    image: NDArray
    strength: float
    fit: bool
    mask_source: MaskSource
    mask_prompt: str
    confidence: float

@dataclass
class DepthToImage(Task):
    depth: NDArray | None
    image: NDArray | None
    strength: float

@dataclass
class Outpaint(Task):
    image: NDArray
    origin: Tuple[int, int]