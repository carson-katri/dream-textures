from dataclasses import dataclass

@dataclass
class Prompt:
    positive: str
    negative: str | None