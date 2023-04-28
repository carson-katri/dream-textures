import bpy
from ..property_groups.dream_prompt import DreamPrompt
from typing import Callable, List
from .generation_result import GenerationResult
from .model import Model

StepCallback = Callable[[], GenerationResult]
Callback = Callable[[], List[GenerationResult] | Exception]

class Backend(bpy.types.PropertyGroup):
    """A backend for Dream Textures.

    Provide the following methods to create a valid backend.

    ```python
    def list_models(self) -> List[Model]
    def generate(self, prompt: DreamPrompt, step_callback: StepCallback, callback: Callback)
    ```
    """

    def list_models(self) -> List[Model]:
        """Provide a list of available models.
        
        The `id` of the model will be provided 
        """
        ...
    
    def draw_prompt(self, layout, context):
        """Draw additional UI in the 'Prompt' panel"""
        ...
    
    def draw_speed_optimzations(self, layout, context):
        """Draw additional UI in the 'Speed Optimizations' panel"""
        ...
    
    def draw_memory_optimzations(self, layout, context):
        """Draw additional UI in the 'Memory Optimizations' panel"""
        ...
    
    def draw_extra(self, layout, context):
        """Draw additional UI in the panel"""
        ...

    def generate(
        self,
        prompt: DreamPrompt,
        step_callback: StepCallback,
        callback: Callback
    ):
        """A request to generate an image.
        
        Use the `DreamPrompt` to get all of the arguments for the generation.
        Call `step_callback` at each step as needed.
        Call `callback` when the generation is complete.

        `DreamPrompt` has several helper functions to access generation options.

        ```python
        prompt.generate_prompt() # get the full prompt string
        prompt.get_seed() # an `int` or `None` (in which case you should provide a random seed yourself).
        prompt.get_optimizations() # creates an `Optimization` type.
        ```

        After collecting the necessary arguments, generate an image in the background and call `step_callback` and `callback` with the results.
        
        > Generation should happen on a separate thread or process, as this method is called from the main thread and will block Blender's UI.

        ```python
        call_my_api(
            prompt=prompt.generate_prompt(),
            seed=prompt.get_seed(),
            on_step=lambda res: callback(GenerationResult(res.image, res.seed)),
            on_response=lambda res: callback([GenerationResult(res.image, res.seed)])
        )
        ```
        """
        ...