from typing import Union, Generator, Callable, List, Optional
import os
from contextlib import nullcontext

from numpy.typing import NDArray
import numpy as np
import random
from .prompt_to_image import Checkpoint, Scheduler, Optimizations, StepPreviewMode, step_latents, step_images, _configure_model_padding
from ...api.models.seamless_axes import SeamlessAxes
from ..future import Future
from ...image_utils import image_to_np, size, resize, ImageOrPath


def image_to_image(
    self,
    
    model: str | Checkpoint,

    scheduler: str | Scheduler,

    optimizations: Optimizations,

    image: ImageOrPath,
    fit: bool,
    strength: float,
    prompt: str | list[str],
    steps: int,
    width: int | None,
    height: int | None,
    seed: int,

    cfg_scale: float,
    use_negative_prompt: bool,
    negative_prompt: str,
    
    seamless_axes: SeamlessAxes | str | bool | tuple[bool, bool] | None,

    step_preview_mode: StepPreviewMode,

    # Stability SDK
    key: str | None = None,

    **kwargs
) -> Generator[Future, None, None]:
    future = Future()
    yield future

    import diffusers
    import torch
    
    device = self.choose_device(optimizations)

    # Stable Diffusion pipeline w/ caching
    pipe = self.load_model(diffusers.AutoPipelineForImage2Image, model, optimizations, scheduler)

    # Optimizations
    pipe = optimizations.apply(pipe, device)

    # RNG
    batch_size = len(prompt) if isinstance(prompt, list) else 1
    generator = []
    for _ in range(batch_size):
        gen = torch.Generator(device="cpu" if device in ("mps", "dml") else device) # MPS and DML do not support the `Generator` API
        generator.append(gen.manual_seed(random.randrange(0, np.iinfo(np.uint32).max) if seed is None else seed))
    if batch_size == 1:
        # Some schedulers don't handle a list of generators: https://github.com/huggingface/diffusers/issues/1909
        generator = generator[0]

    # Init Image
    image = image_to_np(image, mode="RGB")
    if fit:
        height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
        width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor
        image = resize(image, (width, height))
    else:
        width, height = size(image)
    
    # Seamless
    if seamless_axes == SeamlessAxes.AUTO:
        seamless_axes = self.detect_seamless(image)
    _configure_model_padding(pipe.unet, seamless_axes)
    _configure_model_padding(pipe.vae, seamless_axes)

    # Inference
    with torch.inference_mode() if device not in ('mps', "dml") else nullcontext():
        def callback(pipe, step, timestep, callback_kwargs):
            if future.check_cancelled():
                raise InterruptedError()
            future.add_response(step_latents(pipe, step_preview_mode, callback_kwargs["latents"], generator, step, steps))
            return callback_kwargs
        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if use_negative_prompt else None,
                image=[image] * batch_size,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
                callback_on_step_end=callback,
                callback_steps=1,
                output_type="np"
            )
            future.add_response(step_images(result.images, generator, steps, steps))
        except InterruptedError:
            pass
    
    future.set_done()