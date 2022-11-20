from typing import Optional, Annotated, Union, _AnnotatedAlias
import enum
import os
from dataclasses import dataclass

class Pipeline(enum.IntEnum):
    STABLE_DIFFUSION = 0

    STABILITY_SDK = 1

@dataclass(eq=True)
class Optimizations:
    attention_slicing = True
    attention_slice_size: Union[str, int] = "auto"
    cudnn_benchmark: Annotated[bool, "cuda"] = False
    tf32: Annotated[bool, "cuda"] = False
    amp: Annotated[bool, "cuda"] = False
    half_precision: Annotated[bool, "cuda"] = True
    sequential_cpu_offload = False
    channels_last_memory_format = False
    xformers_attention = False

    def can_use(self, property, device) -> bool:
        if not getattr(self, property):
            return False
        if isinstance(getattr(self.__annotations__, property, None), _AnnotatedAlias):
            annotation: _AnnotatedAlias = self.__annotations__[property]
            return annotation.__metadata__ != device
        return True

def _choose_device():
    """
    Automatically select which PyTorch device to use.
    """
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def prompt_to_image(
    self,
    pipeline: Pipeline,
    
    model: str,

    prompt: str,
    steps: int,
    width: int,
    height: int,

    optimizations: Optimizations,

    **kwargs
) -> Optional[bytes]:
    match pipeline:
        case Pipeline.STABLE_DIFFUSION:
            import diffusers
            import torch
            from ...absolute_path import WEIGHTS_PATH

            device = _choose_device()
            
            if optimizations.can_use("cudnn_benchmark", device):
                torch.backends.cudnn.benchmark = True
            
            if optimizations.can_use("tf32", device):
                torch.backends.cuda.matmul.allow_tf32 = True

            if hasattr(self, "_cached_pipe") and self._cached_pipe[1] == optimizations:
                pipe = self._cached_pipe[0]
            else:
                pipe = diffusers.StableDiffusionPipeline.from_pretrained(
                    os.path.join(WEIGHTS_PATH, model),
                    torch_dtype=torch.float16 if optimizations.can_use("half_precision", device) else torch.float32,
                )
                pipe = pipe.to(device)

                if optimizations.can_use("attention_slicing", device):
                    pipe.enable_attention_slicing(optimizations.attention_slice_size)
                
                if optimizations.can_use("sequential_cpu_offload", device):
                    pipe.enable_sequential_cpu_offload()
                
                if optimizations.can_use("channels_last_memory_format", device):
                    pipe.unet.to(memory_format=torch.channels_last)

                if optimizations.can_use("xformers_attention", device):
                    pipe.enable_xformers_memory_efficient_attention()

                setattr(self, "_cached_pipe", (pipe, optimizations))

            if device == "mps":
                # First-time "warmup" pass (necessary on MPS as of diffusers 0.7.2)
                _ = pipe(prompt, num_inference_steps=1)
            
            with torch.inference_mode(mode=False), torch.autocast(device, enabled=optimizations.can_use("amp", device)):
                return pipe(
                    prompt,
                    num_inference_steps=steps,
                    width=width,
                    height=height
                ).images[0]
        case Pipeline.STABILITY_SDK:
            import stability_sdk
        case _:
            raise Exception(f"Unsupported pipeline {pipeline}.")