from typing import Optional, Annotated, Union, _AnnotatedAlias
import enum
import os
from dataclasses import dataclass
from contextlib import nullcontext
import numpy as np
from numpy.typing import NDArray
from concurrent.futures import Future
import threading

class Pipeline(enum.IntEnum):
    STABLE_DIFFUSION = 0

    STABILITY_SDK = 1

class Scheduler(enum.Enum):
    LMS_DISCRETE = "LMS Discrete"
    DDIM = "DDIM"
    PNDM = "PNDM"
    DDPM = "DDPM"
    DPM_SOLVER_MULTISTEP = "DPM Solver Multistep"
    EULER_DISCRETE = "Euler Discrete"
    EULER_ANCESTRAL_DISCRETE = "Euler Ancestral Discrete"

    def create(self, pipeline, pretrained):
        import diffusers
        def scheduler_class():
            match self:
                case Scheduler.LMS_DISCRETE:
                    return diffusers.LMSDiscreteScheduler
                case Scheduler.DDIM:
                    return diffusers.DDIMScheduler
                case Scheduler.PNDM:
                    return diffusers.PNDMScheduler
                case Scheduler.DDPM:
                    return diffusers.DDPMScheduler
                case Scheduler.DPM_SOLVER_MULTISTEP:
                    return diffusers.DPMSolverMultistepScheduler
                case Scheduler.EULER_DISCRETE:
                    return diffusers.EulerDiscreteScheduler
                case Scheduler.EULER_ANCESTRAL_DISCRETE:
                    return diffusers.EulerAncestralDiscreteScheduler
        if pretrained is not None:
            return scheduler_class().from_pretrained(pretrained['model_path'], subfolder=pretrained['subfolder'])
        else:
            return scheduler_class().from_config(pipeline.scheduler.config)
        

@dataclass(eq=True)
class Optimizations:
    attention_slicing: bool = True
    attention_slice_size: Union[str, int] = "auto"
    inference_mode: Annotated[bool, "cuda"] = True
    cudnn_benchmark: Annotated[bool, "cuda"] = False
    tf32: Annotated[bool, "cuda"] = False
    amp: Annotated[bool, "cuda"] = False
    half_precision: Annotated[bool, "cuda"] = True
    sequential_cpu_offload: bool = False
    channels_last_memory_format: bool = False
    # xformers_attention: bool = False # FIXME: xFormers is not currently supported due to a lack of official Windows binaries.

    cpu_only: bool = False

    def can_use(self, property, device) -> bool:
        if not getattr(self, property):
            return False
        if isinstance(self.__annotations__.get(property, None), _AnnotatedAlias):
            annotation: _AnnotatedAlias = self.__annotations__[property]
            return annotation.__metadata__ == device
        return True

def choose_device(self) -> str:
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

class Cancellable:
    def __init__(self, iterable, future: Future):
        self.iterable = iterable
        self.future = future

    def __iter__(self):
        for i in self.iterable:
            yield i
            if self.future.cancelled():
                return

def prompt_to_image(
    self,
    pipeline: Pipeline,
    
    model: str,

    scheduler: Scheduler,

    optimizations: Optimizations,

    prompt: str,
    steps: int,
    width: int,
    height: int,
    seed: int,

    seamless: bool,
    seamless_axes: list[str],

    **kwargs
) -> Future:
    future = Future()

    match pipeline:
        case Pipeline.STABLE_DIFFUSION:
            import diffusers
            import torch
            from PIL import ImageOps
            from ...absolute_path import WEIGHTS_PATH

            # Top level configuration
            if optimizations.cpu_only:
                device = "cpu"
            else:
                device = self.choose_device()
            optimizations.can_use("amp", device)
            
            if optimizations.can_use("cudnn_benchmark", device):
                torch.backends.cudnn.benchmark = True
            
            if optimizations.can_use("tf32", device):
                torch.backends.cuda.matmul.allow_tf32 = True

            # StableDiffusionPipeline w/ caching
            if hasattr(self, "_cached_pipe") and self._cached_pipe[1] == (model, scheduler, optimizations):
                pipe = self._cached_pipe[0]
            else:
                storage_folder = os.path.join(WEIGHTS_PATH, model)
                revision = "main"
                ref_path = os.path.join(storage_folder, "refs", revision)
                with open(ref_path) as f:
                    commit_hash = f.read()

                snapshot_folder = os.path.join(storage_folder, "snapshots", commit_hash)
                pipe = diffusers.StableDiffusionPipeline.from_pretrained(
                    snapshot_folder,
                    torch_dtype=torch.float16 if optimizations.can_use("half_precision", device) else torch.float32,
                )
                is_stable_diffusion_2 = 'stabilityai--stable-diffusion-2' in snapshot_folder
                pipe.scheduler = (Scheduler.EULER_DISCRETE if is_stable_diffusion_2 else scheduler).create(pipe, {
                    'model_path': snapshot_folder,
                    'subfolder': 'scheduler',
                } if is_stable_diffusion_2 else None)

                pipe = pipe.to(device)

                if optimizations.can_use("attention_slicing", device):
                    pipe.enable_attention_slicing(optimizations.attention_slice_size)
                
                if optimizations.can_use("sequential_cpu_offload", device):
                    pipe.enable_sequential_cpu_offload()
                
                if optimizations.can_use("channels_last_memory_format", device):
                    pipe.unet.to(memory_format=torch.channels_last)

                # FIXME: xFormers is not currently supported due to a lack of official Windows binaries.
                # if optimizations.can_use("xformers_attention", device):
                #     pipe.enable_xformers_memory_efficient_attention()

                setattr(self, "_cached_pipe", (pipe, (model, scheduler, optimizations)))

                # First-time "warmup" pass (necessary on MPS as of diffusers 0.7.2)
                if device == "mps":
                    _ = pipe(prompt, num_inference_steps=1)

            # RNG
            generator = None if seed is None else (torch.manual_seed(seed) if device == "mps" else torch.Generator(device=device).manual_seed(seed))
            
            # Seamless
            _configure_model_padding(pipe.unet, seamless, seamless_axes)
            _configure_model_padding(pipe.vae, seamless, seamless_axes)

            # Cancellation
            pipe.progress_bar = lambda iterable: Cancellable(iterable, future)

            # Inference
            def inference():
                with (torch.inference_mode() if optimizations.can_use("inference_mode", device) else nullcontext()), \
                        (torch.autocast(device) if optimizations.can_use("amp", device) else nullcontext()):
                        i = pipe(
                            prompt=prompt,
                            height=height,
                            width=width,
                            num_inference_steps=steps,
                            guidance_scale=7.5,
                            negative_prompt=None,
                            num_images_per_prompt=1,
                            eta=0.0,
                            generator=generator,
                            latents=None,
                            output_type="pil",
                            return_dict=True,
                            callback=None,
                            callback_steps=1
                        ).images[0]
                        if not future.cancelled():
                            future.set_result(np.asarray(ImageOps.flip(i).convert('RGBA'), dtype=np.float32) / 255.)
            t = threading.Thread(target=inference, daemon=True)
            t.start()
        case Pipeline.STABILITY_SDK:
            import stability_sdk
            raise NotImplementedError()
        case _:
            raise Exception(f"Unsupported pipeline {pipeline}.")
    
    return future

def _conv_forward_asymmetric(self, input, weight, bias):
    import torch.nn as nn
    """
    Patch for Conv2d._conv_forward that supports asymmetric padding
    """
    working = nn.functional.pad(input, self.asymmetric_padding[0], mode=self.asymmetric_padding_mode[0])
    working = nn.functional.pad(working, self.asymmetric_padding[1], mode=self.asymmetric_padding_mode[1])
    return nn.functional.conv2d(working, weight, bias, self.stride, nn.modules.utils._pair(0), self.dilation, self.groups)

def _configure_model_padding(model, seamless, seamless_axes):
    import torch.nn as nn
    """
    Modifies the 2D convolution layers to use a circular padding mode based on the `seamless` and `seamless_axes` options.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if seamless:
                m.asymmetric_padding_mode = (
                    'circular' if ('x' in seamless_axes) else 'constant',
                    'circular' if ('y' in seamless_axes) else 'constant'
                )
                m.asymmetric_padding = (
                    (m._reversed_padding_repeated_twice[0], m._reversed_padding_repeated_twice[1], 0, 0),
                    (0, 0, m._reversed_padding_repeated_twice[2], m._reversed_padding_repeated_twice[3])
                )
                m._conv_forward = _conv_forward_asymmetric.__get__(m, nn.Conv2d)
            else:
                m._conv_forward = nn.Conv2d._conv_forward.__get__(m, nn.Conv2d)
                if hasattr(m, 'asymmetric_padding_mode'):
                    del m.asymmetric_padding_mode
                if hasattr(m, 'asymmetric_padding'):
                    del m.asymmetric_padding