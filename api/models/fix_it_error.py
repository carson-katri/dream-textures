from typing import Callable, Any
from .generation_arguments import GenerationArguments
from dataclasses import dataclass

class FixItError(Exception):
    """An exception with a solution.

    Call the `draw` method to render the UI elements responsible for resolving this error.
    """
    def __init__(self, message, solution: 'Solution'):
        super().__init__(message)

        self._solution = solution
    
    def _draw(self, dream_prompt, context, layout):
        self._solution._draw(dream_prompt, context, layout)
    
    @dataclass
    class Solution:
        def _draw(self, dream_prompt, context, layout):
            ...

    @dataclass
    class ChangeProperty(Solution):
        """Prompts the user to change the given `property` of the `GenerationArguments`."""
        property: str

        def _draw(self, dream_prompt, context, layout):
            layout.prop(dream_prompt, self.property)
    
    @dataclass
    class RunOperator(Solution):
        """Runs the given operator"""
        title: str
        operator: str
        modify_operator: Callable[[Any], None]

        def _draw(self, dream_prompt, context, layout):
            self.modify_operator(
                layout.operator(self.operator, text=self.title)
            )