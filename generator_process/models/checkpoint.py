from dataclasses import dataclass

from .model_config import ModelConfig


@dataclass(frozen=True)
class Checkpoint:
    path: str
    config: ModelConfig | str | None
