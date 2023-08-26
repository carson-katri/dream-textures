from dataclasses import dataclass

from .model_config import ModelConfig


@dataclass
class Checkpoint:
    path: str
    config: ModelConfig | str | None
