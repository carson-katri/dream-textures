import numpy as np
from .prompt_to_image import Optimizations, Scheduler, StepPreviewMode, _configure_model_padding
from .detect_seamless import SeamlessAxes
import random
from dataclasses import dataclass
from numpy.typing import NDArray
from ..models.upscale_tiler import UpscaleTiler

@dataclass
class ImageUpscaleResult:
    image: NDArray | None
    tile: int
    total: int
    final: bool


def upscale(
    self,
    image: NDArray,
    
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
    from PIL import Image, ImageOps
    import torch
    import diffusers

    if optimizations.cpu_only:
        device = "cpu"
    else:
        device = self.choose_device()

    pipe = diffusers.StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        revision="fp16" if optimizations.can_use_half(device) else None,
        torch_dtype=torch.float16 if optimizations.can_use_half(device) else torch.float32
    )
    pipe.scheduler = scheduler.create(pipe, None)
    # vae would automatically be made float32 within the pipeline, but it fails to convert after offloading is enabled
    pipe.vae.to(dtype=torch.float32)
    if optimizations.can_use_cpu_offload(device) == "off":
        pipe = pipe.to(device)
    pipe = optimizations.apply(pipe, device)

    generator = torch.Generator(device="cpu" if device in ("mps", "privateuseone") else device) # MPS and DML do not support the `Generator` API
    if seed is None:
        seed = random.randrange(0, np.iinfo(np.uint32).max)

    if image.shape[2] == 4:
        image = image[:, :, :3]
    tiler = UpscaleTiler(image, 4, tile_size, blend, seamless_axes)
    _configure_model_padding(pipe.unet, seamless_axes & ~tiler.seamless_axes)
    _configure_model_padding(pipe.vae, seamless_axes & ~tiler.seamless_axes)
    for i in range(0, len(tiler), optimizations.batch_size):
        batch_size = min(len(tiler)-i, optimizations.batch_size)
        ids = list(range(i, i+batch_size))
        low_res_tiles = [Image.fromarray(tiler[id]) for id in ids]
        high_res_tiles = pipe(
            prompt=[prompt] * batch_size,
            image=low_res_tiles,
            num_inference_steps=steps,
            generator=generator.manual_seed(seed),
            guidance_scale=cfg_scale,
        ).images

        # not implemented in diffusers.StableDiffusionUpscalePipeline
        # Offload last model to CPU
        if hasattr(pipe, "final_offload_hook") and pipe.final_offload_hook is not None:
            pipe.final_offload_hook.offload()

        for id, tile in zip(ids, high_res_tiles):
            tiler[id] = np.array(tile)
        step = None
        if step_preview_mode != StepPreviewMode.NONE:
            step = Image.fromarray(tiler.combined().astype(np.uint8))
        yield ImageUpscaleResult(
            (np.asarray(ImageOps.flip(step).convert('RGBA'), dtype=np.float32) / 255.) if step is not None else None,
            i + batch_size,
            len(tiler),
            (i + batch_size) == len(tiler)
        )
    if step_preview_mode == StepPreviewMode.NONE:
        final = Image.fromarray(tiler.combined().astype(np.uint8))
        yield ImageUpscaleResult(
            np.asarray(ImageOps.flip(final).convert('RGBA'), dtype=np.float32) / 255.,
            len(tiler),
            len(tiler),
            True
        )
