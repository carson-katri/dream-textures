from typing import Union, Generator, Callable, List, Optional, Any
import os
from contextlib import nullcontext

import numpy as np
import random
from ...api.models.seamless_axes import SeamlessAxes
from ...api.models.step_preview_mode import StepPreviewMode
from ..models.scheduler import Scheduler
from ..models.optimizations import Optimizations
from ..models.image_generation_result import ImageGenerationResult
from ..future import Future

class CachedPipeline:
    """A pipeline that has been cached for subsequent runs."""

    pipeline: Any
    """The diffusers pipeline to re-use"""

    invalidation_properties: tuple
    """Values that, when changed, will invalid this cached pipeline"""

    snapshot_folder: str
    """The snapshot folder containing the model"""

    def __init__(self, pipeline: Any, invalidation_properties: tuple, snapshot_folder: str):
        self.pipeline = pipeline
        self.invalidation_properties = invalidation_properties
        self.snapshot_folder = snapshot_folder

    def is_valid(self, properties: tuple):
        return properties == self.invalidation_properties

def load_pipe(self, action, generator_pipeline, model, optimizations, scheduler, device, **kwargs):
    """
    Use a cached pipeline, or create the pipeline class and cache it.
    
    The cached pipeline will be invalidated if the model or use_cpu_offload options change.
    """
    import torch
    import gc

    invalidation_properties = (
        action, model, device,
        optimizations.can_use_cpu_offload(device),
        optimizations.can_use("half_precision", device),
    )
    cached_pipe: CachedPipeline = self._cached_pipe if hasattr(self, "_cached_pipe") else None
    if cached_pipe is not None and cached_pipe.is_valid(invalidation_properties):
        pipe = cached_pipe.pipeline
    else:
        # Release the cached pipe before loading the new one.
        if cached_pipe is not None:
            del self._cached_pipe
            del cached_pipe
            gc.collect()

        variant = "fp16" if optimizations.can_use_half(device) else None
        snapshot_folder = model_snapshot_folder(model, variant)
        pipe = generator_pipeline.from_pretrained(
            snapshot_folder,
            variant="fp16",
            torch_dtype=torch.float16 if optimizations.can_use_half(device) else torch.float32,
            **kwargs
        )
        if optimizations.can_use_cpu_offload(device) == "off":
            pipe = pipe.to(device)
        setattr(self, "_cached_pipe", CachedPipeline(pipe, invalidation_properties, snapshot_folder))
        cached_pipe = self._cached_pipe
    if scheduler is not None:
        if 'scheduler' in os.listdir(cached_pipe.snapshot_folder):
            pipe.scheduler = scheduler.create(pipe, {
                'model_path': cached_pipe.snapshot_folder,
                'subfolder': 'scheduler',
            })
        else:
            pipe.scheduler = scheduler.create(pipe, None)
    return pipe

def choose_device(self) -> str:
    """
    Automatically select which PyTorch device to use.
    """
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    if Pipeline.directml_available():
        import torch_directml
        if torch_directml.is_available():
            torch.utils.rename_privateuse1_backend("dml")
            return "dml"
    return "cpu"

def model_snapshot_folder(model, preferred_revision: str | None = None):
    """ Try to find the preferred revision, but fallback to another revision if necessary. """
    import diffusers
    storage_folder = os.path.join(diffusers.utils.DIFFUSERS_CACHE, model)
    if not os.path.exists(os.path.join(storage_folder, "refs")):
        storage_folder = os.path.join(diffusers.utils.hub_utils.old_diffusers_cache, model)
    if os.path.exists(os.path.join(storage_folder, 'model_index.json')): # converted model
        snapshot_folder = storage_folder
    else: # hub model
        revisions = {}
        for revision in os.listdir(os.path.join(storage_folder, "refs")):
            ref_path = os.path.join(storage_folder, "refs", revision)
            with open(ref_path) as f:
                commit_hash = f.read()

            snapshot_folder = os.path.join(storage_folder, "snapshots", commit_hash)
            if len(os.listdir(snapshot_folder)) > 1:
                revisions[revision] = snapshot_folder

        if len(revisions) == 0:
            return None
        elif preferred_revision in revisions:
            revision = preferred_revision
        elif preferred_revision in [None, "fp16"] and "main" in revisions:
            revision = "main"
        elif preferred_revision in [None, "main"] and "fp16" in revisions:
            revision = "fp16"
        else:
            revision = next(iter(revisions.keys()))
        snapshot_folder = revisions[revision]

    return snapshot_folder

def use_sdxl(model, optimizations, device):
    import json
    snapshot_folder = model_snapshot_folder(model, "fp16" if optimizations.can_use_half(device) else None)
    if snapshot_folder is None:
        return False
    model_index = os.path.join(snapshot_folder, "model_index.json")
    if not os.path.exists(model_index):
        return False
    with open(model_index) as f:
        return json.load(f)["_class_name"] == "StableDiffusionXLPipeline"

def prompt_to_image(
    self,
    
    model: str,

    scheduler: Scheduler,

    optimizations: Optimizations,

    prompt: str | list[str],
    steps: int,
    width: int | None,
    height: int | None,
    seed: int,

    cfg_scale: float,
    use_negative_prompt: bool,
    negative_prompt: str,
    
    seamless_axes: SeamlessAxes | str | bool | tuple[bool, bool] | None,

    iterations: int,

    step_preview_mode: StepPreviewMode,

    # Stability SDK
    key: str | None = None,

    **kwargs
) -> Generator[Future, None, None]:
    future = Future()
    yield future

    import diffusers
    import torch
    from PIL import Image, ImageOps

    if optimizations.cpu_only:
        device = "cpu"
    else:
        device = self.choose_device()

    # Stable Diffusion pipeline w/ caching
    if use_sdxl(model, optimizations, device):
        pipe = load_pipe(self, "prompt", diffusers.StableDiffusionXLPipeline, model, optimizations, scheduler, device)
    else:
        pipe = load_pipe(self, "prompt", diffusers.StableDiffusionPipeline, model, optimizations, scheduler, device)
    height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor

    # Optimizations
    pipe: diffusers.StableDiffusionPipeline = optimizations.apply(pipe, device)

    # RNG
    batch_size = len(prompt) if isinstance(prompt, list) else 1
    generator = []
    for _ in range(batch_size):
        gen = torch.Generator(device="cpu" if device in ("mps", "dml") else device) # MPS and DML do not support the `Generator` API
        generator.append(gen.manual_seed(random.randrange(0, np.iinfo(np.uint32).max) if seed is None else seed))
    if batch_size == 1:
        # Some schedulers don't handle a list of generators: https://github.com/huggingface/diffusers/issues/1909
        generator = generator[0]
    
    # Seamless
    _configure_model_padding(pipe.unet, seamless_axes)
    _configure_model_padding(pipe.vae, seamless_axes)

    # Inference
    with torch.inference_mode() if device not in ('mps', "dml") else nullcontext():
        def callback(step, timestep, latents):
            future.add_response(ImageGenerationResult.step_preview(self, step_preview_mode, width, height, latents, generator, step))
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            negative_prompt=negative_prompt if use_negative_prompt else None,
            num_images_per_prompt=1,
            eta=0.0,
            generator=generator,
            latents=None,
            output_type="pil",
            return_dict=True,
            callback=callback,
            callback_steps=1,
            #cfg_end=optimizations.cfg_end
        )
        future.add_response(ImageGenerationResult(
            [np.asarray(ImageOps.flip(image).convert('RGBA'), dtype=np.float32) / 255.
                for image in result.images],
            [gen.initial_seed() for gen in generator] if isinstance(generator, list) else [generator.initial_seed()],
            steps,
            True
        ))
    
    future.set_done()

def _conv_forward_asymmetric(self, input, weight, bias):
    import torch.nn as nn
    """
    Patch for Conv2d._conv_forward that supports asymmetric padding
    """
    if input.device.type == "dml":
        # DML pad() will wrongly fill the tensor in constant mode with the supplied value
        # (default 0) when padding on both ends of a dimension, can't split to two calls.
        working = nn.functional.pad(input, self._reversed_padding_repeated_twice, mode='circular')
        pad_w0, pad_w1, pad_h0, pad_h1 = self._reversed_padding_repeated_twice
        if self.asymmetric_padding_mode[0] == 'constant':
            working[:, :, :, :pad_w0] = 0
            if pad_w1 > 0:
                working[:, :, :, -pad_w1:] = 0
        if self.asymmetric_padding_mode[1] == 'constant':
            working[:, :, :pad_h0] = 0
            if pad_h1 > 0:
                working[:, :, -pad_h1:] = 0
    else:
        working = nn.functional.pad(input, self.asymmetric_padding[0], mode=self.asymmetric_padding_mode[0])
        working = nn.functional.pad(working, self.asymmetric_padding[1], mode=self.asymmetric_padding_mode[1])
    return nn.functional.conv2d(working, weight, bias, self.stride, nn.modules.utils._pair(0), self.dilation, self.groups)

def _configure_model_padding(model, seamless_axes):
    import torch.nn as nn
    """
    Modifies the 2D convolution layers to use a circular padding mode based on the `seamless` and `seamless_axes` options.
    """
    seamless_axes = SeamlessAxes(seamless_axes)
    if seamless_axes == SeamlessAxes.AUTO:
        seamless_axes = seamless_axes.OFF
    if getattr(model, "seamless_axes", SeamlessAxes.OFF) == seamless_axes:
        return
    model.seamless_axes = seamless_axes
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if seamless_axes.x or seamless_axes.y:
                m.asymmetric_padding_mode = (
                    'circular' if seamless_axes.x else 'constant',
                    'circular' if seamless_axes.y else 'constant'
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
