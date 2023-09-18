from dataclasses import dataclass
from typing import Tuple, List
from numpy.typing import NDArray

@dataclass
class ControlNet:
    model: str
    """The selected ControlNet model used for generation"""

    image: NDArray
    """The control image"""

    strength: float
    """The strength of the ControlNet's influence"""