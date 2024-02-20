import numpy as np
from .prompt_to_image import Optimizations, Scheduler, StepPreviewMode, _configure_model_padding
from ...api.models.seamless_axes import SeamlessAxes
import random
from numpy.typing import NDArray
from ..models import Checkpoint, Optimizations, Scheduler, UpscaleTiler, step_images
from ..future import Future
from contextlib import nullcontext
from ...image_utils import rgb, rgba

def upscale(
    self,
    image: NDArray,

    model: str | Checkpoint,
    
    prompt: str,
    steps: int,
    seed: int,
    cfg_scale: float,
    scheduler: Scheduler,
    
    tile_size: int,
    blend: int,
    seamless_axes: SeamlessAxes | str | bool | tuple[bool, bool] | None,

    optimizations: Optimizations,

    step_preview_mode: StepPreviewMode,

    **kwargs
):
    future = Future()
    yield future

    import torch
    import diffusers

    device = self.choose_device(optimizations)

    pipe = self.load_model(diffusers.StableDiffusionUpscalePipeline, model, optimizations, scheduler)

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
    
    # Seamless
    tiler = UpscaleTiler(image, 4, tile_size, blend, seamless_axes)
    _configure_model_padding(pipe.unet, seamless_axes & ~tiler.seamless_axes)
    _configure_model_padding(pipe.vae, seamless_axes & ~tiler.seamless_axes)

    for i in range(0, len(tiler), optimizations.batch_size):
        if future.check_cancelled():
            future.set_done()
            return
        batch_size = min(len(tiler)-i, optimizations.batch_size)
        ids = list(range(i, i+batch_size))
        low_res_tiles = [rgb(tiler[id]) for id in ids]
        # Inference
        with torch.inference_mode() if device not in ('mps', "dml") else nullcontext():
            high_res_tiles = pipe(
                prompt=[prompt[0] if isinstance(prompt, list) else prompt] * batch_size,
                image=low_res_tiles,
                num_inference_steps=steps,
                generator=generator,
                guidance_scale=cfg_scale,
                output_type="np"
            ).images

        for id, tile in zip(ids, high_res_tiles):
            tiler[id] = rgba(tile)

        if step_preview_mode != StepPreviewMode.NONE:
            future.add_response(step_images(
                [tiler.combined()],
                generator,
                i + batch_size,
                len(tiler),
            ))
    if step_preview_mode == StepPreviewMode.NONE:
        future.add_response(step_images(
            [tiler.combined()],
            generator,
            len(tiler),
            len(tiler)
        ))
    future.set_done()
