try:
    import bpy
    from typing import Callable, List, Tuple
    from ..models.generation_result import GenerationResult
    from ..models.task import Task
    from ..models.model import Model
    from ..models.prompt import Prompt
    from ..models.seamless_axes import SeamlessAxes
    from ..models.step_preview_mode import StepPreviewMode

    StepCallback = Callable[[List[GenerationResult]], None]
    Callback = Callable[[List[GenerationResult] | Exception], None]

    class Backend(bpy.types.PropertyGroup):
        """A backend for Dream Textures.

        Provide the following methods to create a valid backend.

        ```python
        def list_models(self) -> List[Model]
        def generate(
            self,
            task: Task,
            model: Model,
            prompt: Prompt,
            size: Tuple[int, int] | None,
            seamless_axes: SeamlessAxes,

            step_callback: StepCallback,
            callback: Callback
        )
        ```
        """

        @classmethod
        def register(cls):
            from ...property_groups.dream_prompt import DreamPrompt
            setattr(DreamPrompt, cls._attribute(), bpy.props.PointerProperty(type=cls))

        @classmethod
        def _id(cls) -> str:
            return f"{cls.__module__}.{cls.__name__}"
        
        @classmethod
        def _attribute(cls) -> str:
            return cls._id().replace('.', '_')
        
        @classmethod
        def _lookup(cls, id):
            return next(backend for backend in cls._list_backends() if backend._id() == id)
        
        @classmethod
        def _list_backends(cls):
            return cls.__subclasses__()

        def list_models(self, context) -> List[Model]:
            """Provide a list of available models.
            
            The `id` of the model will be provided 
            """
            ...
        
        def list_schedulers(self, context) -> List[str]:
            """Provide a list of available schedulers."""
            ...
        
        def draw_prompt(self, layout, context):
            """Draw additional UI in the 'Prompt' panel"""
            ...
        
        def draw_advanced(self, layout, context):
            """Draw additional UI in the 'Advanced' panel"""
            ...

        def draw_speed_optimizations(self, layout, context):
            """Draw additional UI in the 'Speed Optimizations' panel"""
            ...
        
        def draw_memory_optimizations(self, layout, context):
            """Draw additional UI in the 'Memory Optimizations' panel"""
            ...
        
        def draw_extra(self, layout, context):
            """Draw additional UI in the panel"""
            ...

        def generate(
            self,
            task: Task,
            model: Model,
            prompt: Prompt,
            size: Tuple[int, int] | None,
            seed: int,
            steps: int,
            guidance_scale: float,
            scheduler: str,
            seamless_axes: SeamlessAxes,
            step_preview_mode: StepPreviewMode,
            iterations: int,

            step_callback: StepCallback,
            callback: Callback
        ):
            """A request to generate an image."""
            ...
except:
    pass