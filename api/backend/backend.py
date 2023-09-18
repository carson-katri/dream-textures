try:
    import bpy
    from typing import Callable, List, Tuple
    from ..models.generation_arguments import GenerationArguments
    from ..models.generation_result import GenerationResult
    from ..models.model import Model

    StepCallback = Callable[[List[GenerationResult]], bool]
    Callback = Callable[[List[GenerationResult] | Exception], None]

    class Backend(bpy.types.PropertyGroup):
        """A backend for Dream Textures.

        Provide the following methods to create a valid backend.

        ```python
        def list_models(self) -> List[Model]
        def generate(
            self,
            arguments: GenerationArguments,

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
        def unregister(cls):
            from ...property_groups.dream_prompt import DreamPrompt
            delattr(DreamPrompt, cls._attribute())

        @classmethod
        def _id(cls) -> str:
            return f"{cls.__module__}.{cls.__name__}"
        
        @classmethod
        def _attribute(cls) -> str:
            return cls._id().replace('.', '_')
        
        @classmethod
        def _lookup(cls, id):
            return next(
                (backend for backend in cls._list_backends() if backend._id() == id),
                next(iter(cls._list_backends()), None)
            )
        
        @classmethod
        def _list_backends(cls):
            return cls.__subclasses__()

        def list_models(self, context) -> List[Model]:
            """Provide a list of available models.
            
            The `id` of the model will be provided.
            """
            ...
        
        def list_controlnet_models(self, context) -> List[Model]:
            """Provide a list of available ControlNet models.

            The `id` of the model will be provided.
            """
            return []
        
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
        
        def get_batch_size(self, context) -> int:
            """Return the selected batch size for the backend (if applicable).
            
            A default implementation is provided that returns `1`.
            """
            return 1

        def generate(
            self,
            arguments: GenerationArguments,
            step_callback: StepCallback,
            callback: Callback
        ):
            """
            A request to generate an image.

            If the `step_callback` returns `False`, the generation should be cancelled.
            After cancelling, `callback` should be called with an `InterruptedError`.
            """
            ...
        
        def validate(
            self,
            arguments: GenerationArguments
        ):
            """Validates the given arguments in the UI without generating.
            
            This validation should occur as quickly as possible.
            
            To report problems with the inputs, raise a `ValueError`.
            Use the `FixItError` to provide a solution to the problem as well.
            
            ```python
            if arguments.steps % 2 == 0:
                throw FixItError(
                    "The number of steps is even",
                    solution=FixItError.UpdateGenerationArgumentsSolution(
                        title="Add 1 more step",
                        arguments=dataclasses.replace(
                            arguments,
                            steps=arguments.steps + 1
                        )
                    )
                )
            ```
            """
            ...
except:
    pass