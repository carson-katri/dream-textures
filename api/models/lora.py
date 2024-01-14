from dataclasses import dataclass

@dataclass
class Lora:
    model: str
    """The selected LoRA model used for generation"""

    weight: float
    """The strength of the LoRA's influence"""