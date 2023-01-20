from typing import Callable, Any

class FixItError(Exception):
    """An exception with a solution.

    Call the `draw` method to render the UI elements responsible for resolving this error.
    """
    def __init__(self, message, fix_it: Callable[[Any, Any], None]):
        super().__init__(message)

        self._fix_it = fix_it
    
    def draw(self, context, layout):
        self._fix_it(context, layout)