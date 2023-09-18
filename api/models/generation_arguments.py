from dataclasses import dataclass
from typing import Tuple, List
from ..models.task import Task
from ..models.model import Model
from ..models.prompt import Prompt
from ..models.seamless_axes import SeamlessAxes
from ..models.step_preview_mode import StepPreviewMode
from ..models.control_net import ControlNet

@dataclass
class GenerationArguments:
    task: Task
    """The type of generation to perform.
    
    Use a match statement to perform different actions based on the selected task.
    
    ```python
    match task:
        case PromptToImage():
            ...
        case ImageToImage(image=image, strength=strength, fit=fit):
            ...
        case Inpaint(image=image, fit=fit, strength=strength, mask_source=mask_source, mask_prompt=mask_prompt, confidence=confidence):
            ...
        case DepthToImage(depth=depth, image=image, strength=strength):
            ...
        case Outpaint(image=image, origin=origin):
            ...
        case _:
            raise NotImplementedError()
    ```
    """

    model: Model
    """The selected model.

    This is one of the options provided by `Backend.list_models`.
    """

    prompt: Prompt
    """The positive and (optionally) negative prompt.

    If `prompt.negative` is `None`, then the 'Negative Prompt' panel was disabled by the user.
    """
    
    size: Tuple[int, int] | None
    """The target size of the image, or `None` to use the native size of the model."""

    seed: int
    """The random or user-provided seed to use."""

    steps: int
    """The number of inference steps to perform."""

    guidance_scale: float
    """The selected classifier-free guidance scale."""

    scheduler: str
    """The selected scheduler.
    
    This is one of the options provided by `Backend.list_schedulers`.
    """

    seamless_axes: SeamlessAxes
    """Which axes to tile seamlessly."""

    step_preview_mode: StepPreviewMode
    """The style of preview to display at each step."""

    iterations: int
    """The number of images to generate.
    
    The value sent to `callback` should contain the same number of `GenerationResult` instances in a list.
    """

    control_nets: List[ControlNet]

    @staticmethod
    def _map_property_name(name: str) -> str | List[str] | None:
        """Converts a property name from `GenerationArguments` to the corresponding property of a `DreamPrompt`."""
        match name:
            case "model":
                return "model"
            case "prompt":
                return ["prompt", "use_negative_prompt", "negative_prompt"]
            case "prompt.positive":
                return "prompt"
            case "prompt.negative":
                return ["use_negative_prompt", "negative_prompt"]
            case "size":
                return ["use_size", "width", "height"]
            case "seed":
                return "seed"
            case "steps":
                return "steps"
            case "guidance_scale":
                return "cfg_scale"
            case "scheduler":
                return "scheduler"
            case "seamless_axes":
                return "seamless_axes"
            case "step_preview_mode":
                return "step_preview_mode"
            case "iterations":
                return "iterations"
            case _:
                return None