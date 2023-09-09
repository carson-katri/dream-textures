import numpy as np
from .prompt_to_image import Optimizations, Scheduler, StepPreviewMode, _configure_model_padding
from ...api.models.seamless_axes import SeamlessAxes
import random
from numpy.typing import NDArray
from ..models import Checkpoint, Optimizations, Scheduler, UpscaleTiler, ImageGenerationResult
from ..future import Future
from contextlib import nullcontext

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

    from PIL import Image, ImageOps
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

    if image.shape[2] == 4:
        image = image[:, :, :3]
    for i in range(0, len(tiler), optimizations.batch_size):
        if future.check_cancelled():
            future.set_done()
            return
        batch_size = min(len(tiler)-i, optimizations.batch_size)
        ids = list(range(i, i+batch_size))
        low_res_tiles = [Image.fromarray(tiler[id]).convert('RGB') for id in ids]
        # Inference
        with torch.inference_mode() if device not in ('mps', "dml") else nullcontext():
            high_res_tiles = pipe(
                prompt=[prompt[0] if isinstance(prompt, list) else prompt] * batch_size,
                image=low_res_tiles,
                num_inference_steps=steps,
                generator=generator,
                guidance_scale=cfg_scale,
            ).images

            # not implemented in diffusers.StableDiffusionUpscalePipeline
            # Offload last model to CPU
            if hasattr(pipe, "final_offload_hook") and pipe.final_offload_hook is not None:
                pipe.final_offload_hook.offload()

        for id, tile in zip(ids, high_res_tiles):
            tiler[id] = np.array(tile.convert('RGBA'))
        step = None
        if step_preview_mode != StepPreviewMode.NONE:
            step = Image.fromarray(tiler.combined().astype(np.uint8))
            future.add_response(ImageGenerationResult(
                [(np.asarray(ImageOps.flip(step).convert('RGBA'), dtype=np.float32) / 255.)],
                [seed],
                i + batch_size,
                (i + batch_size) == len(tiler),
                total=len(tiler)
            ))
    if step_preview_mode == StepPreviewMode.NONE:
        final = Image.fromarray(tiler.combined().astype(np.uint8))
        future.add_response(ImageGenerationResult(
            [np.asarray(ImageOps.flip(final).convert('RGBA'), dtype=np.float32) / 255.],
            [seed],
            len(tiler),
            True,
            total=len(tiler)
        ))
    future.set_done()
