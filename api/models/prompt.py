from dataclasses import dataclass
from typing import List

@dataclass
class Prompt:
    positive: str | List[str]
    negative: str | List[str] | None