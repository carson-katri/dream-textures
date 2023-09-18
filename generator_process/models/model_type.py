import enum

from ...api.models.task import *
from .model_config import ModelConfig


class ModelType(enum.IntEnum):
    """
    Inferred model type from the U-Net `in_channels`.
    """
    UNKNOWN = 0
    PROMPT_TO_IMAGE = 4
    DEPTH = 5
    UPSCALING = 7
    INPAINTING = 9

    CONTROL_NET = -1
    UNSPECIFIED_CHECKPOINT = -2

    @classmethod
    def _missing_(cls, _):
        return cls.UNKNOWN

    def recommended_model(self) -> str:
        """Provides a recommended model for a given task.

        This method has a bias towards the latest version of official Stability AI models.
        """
        match self:
            case ModelType.PROMPT_TO_IMAGE:
                return "stabilityai/stable-diffusion-2-1"
            case ModelType.DEPTH:
                return "stabilityai/stable-diffusion-2-depth"
            case ModelType.UPSCALING:
                return "stabilityai/stable-diffusion-x4-upscaler"
            case ModelType.INPAINTING:
                return "stabilityai/stable-diffusion-2-inpainting"
            case _:
                return "stabilityai/stable-diffusion-2-1"

    def matches_task(self, task: Task) -> bool:
        """Indicates if the model type is correct for a given `Task`.

        If not an error should be shown to the user to select a different model.
        """
        if self == ModelType.UNSPECIFIED_CHECKPOINT:
            return True
        match task:
            case PromptToImage():
                return self == ModelType.PROMPT_TO_IMAGE
            case Inpaint():
                return self == ModelType.INPAINTING
            case DepthToImage():
                return self == ModelType.DEPTH
            case Outpaint():
                return self == ModelType.INPAINTING
            case ImageToImage():
                return self == ModelType.PROMPT_TO_IMAGE
            case _:
                return False

    @staticmethod
    def from_task(task: Task) -> 'ModelType | None':
        match task:
            case PromptToImage():
                return ModelType.PROMPT_TO_IMAGE
            case Inpaint():
                return ModelType.INPAINTING
            case DepthToImage():
                return ModelType.DEPTH
            case Outpaint():
                return ModelType.INPAINTING
            case ImageToImage():
                return ModelType.PROMPT_TO_IMAGE
            case _:
                return None

    @staticmethod
    def from_config(config: ModelConfig):
        match config:
            case ModelConfig.AUTO_DETECT:
                return ModelType.UNSPECIFIED_CHECKPOINT
            case ModelConfig.STABLE_DIFFUSION_2_DEPTH:
                return ModelType.DEPTH
            case ModelConfig.STABLE_DIFFUSION_2_INPAINTING:
                return ModelType.INPAINTING
            case ModelConfig.CONTROL_NET_1_5 | ModelConfig.CONTROL_NET_2_1:
                return ModelType.CONTROL_NET
            case _:
                return ModelType.PROMPT_TO_IMAGE
