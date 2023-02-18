from .model_type import ModelType
from dataclasses import dataclass

@dataclass
class Model:
    id: str
    author: str
    tags: list[str]
    likes: int
    downloads: int
    model_type: ModelType