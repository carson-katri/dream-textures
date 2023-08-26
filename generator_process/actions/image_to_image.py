from typing import Union, Generator, Callable, List, Optional
import os
from contextlib import nullcontext

from numpy.typing import NDArray
import numpy as np
import random
from .prompt_to_image import Checkpoint, Scheduler, Optimizations, StepPreviewMode, ImageGenerationResult, _configure_model_padding
from ...api.models.seamless_axes import SeamlessAxes
from ..future import Future


def image_to_image(
    self,
    
    model: str | Checkpoint,

    scheduler: Scheduler,

    optimizations: Optimizations,

    image: NDArray,
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
    from PIL import Image, ImageOps
    import PIL.Image
    
    device = self.choose_device(optimizations)

    # Stable Diffusion pipeline w/ caching
    pipe = self.load_model(diffusers.AutoPipelineForImage2Image, model, optimizations)

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
    init_image = Image.fromarray(image).convert('RGB')

    if fit:
        height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
        width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor
        init_image = init_image.resize((width, height))
    else:
        width = init_image.width
        height = init_image.height
    
    # Seamless
    if seamless_axes == SeamlessAxes.AUTO:
        seamless_axes = self.detect_seamless(np.array(init_image) / 255)
    _configure_model_padding(pipe.unet, seamless_axes)
    _configure_model_padding(pipe.vae, seamless_axes)

    # Inference
    with torch.inference_mode() if device not in ('mps', "dml") else nullcontext():
        def callback(step, timestep, latents):
            future.add_response(ImageGenerationResult.step_preview(self, step_preview_mode, width, height, latents, generator, step))
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if use_negative_prompt else None,
            image=[init_image] * batch_size,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            generator=generator,
            callback=callback
        )
        future.add_response(ImageGenerationResult(
            [np.asarray(ImageOps.flip(image).convert('RGBA'), dtype=np.float32) / 255.
                for image in result.images],
            [gen.initial_seed() for gen in generator] if isinstance(generator, list) else [generator.initial_seed()],
            steps,
            True
        ))
    
    future.set_done()