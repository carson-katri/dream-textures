import numpy as np
from .prompt_to_image import Optimizations, Scheduler, StepPreviewMode
import random
from dataclasses import dataclass
from numpy.typing import NDArray

@dataclass
class ImageUpscaleResult:
    image: NDArray | None
    tile: int
    total: int
    final: bool

def upscale(
    self,
    image: str,
    
    prompt: str,
    steps: int,
    seed: int,
    cfg_scale: float,
    scheduler: Scheduler,
    
    tile_size: int,
    blend: int,

    optimizations: Optimizations,

    step_preview_mode: StepPreviewMode,

    **kwargs
):
    from PIL import Image, ImageOps
    import torch
    import diffusers
    from tiler import Tiler, Merger

    if optimizations.cpu_only:
        device = "cpu"
    else:
        device = self.choose_device()

    pipe = diffusers.StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        revision="fp16" if optimizations.can_use("half_precision", device) else None,
        torch_dtype=torch.float16 if optimizations.can_use("half_precision", device) else torch.float32
    )
    pipe.scheduler = scheduler.create(pipe, None)
    pipe = pipe.to(device)
    pipe = optimizations.apply(pipe, device)

    low_res_image = Image.open(image)

    generator = torch.Generator(device="cpu" if device == "mps" else device) # MPS does not support the `Generator` API
    if seed is None:
        seed = random.randrange(0, np.iinfo(np.uint32).max)

    tiler = Tiler(
        data_shape=(low_res_image.size[0], low_res_image.size[1], len(low_res_image.getbands())),
        tile_shape=(tile_size, tile_size, len(low_res_image.getbands())),
        overlap=(blend, blend, 0),
        channel_dimension=2
    )
    merger = Merger(Tiler(
        data_shape=(low_res_image.size[0] * 4, low_res_image.size[1] * 4, 3),
        tile_shape=(tile_size * 4, tile_size * 4, 3),
        overlap=(blend * 4, blend * 4, 0),
        channel_dimension=2
    ))
    input_array = np.array(low_res_image)
    for id, tile in tiler(input_array, progress_bar=True):
        merger.add(id, np.array(pipe(
            prompt=prompt,
            image=Image.fromarray(tile),
            num_inference_steps=steps,
            generator=generator.manual_seed(seed),
            guidance_scale=cfg_scale,
        ).images[0]))
        if step_preview_mode != StepPreviewMode.NONE:
            step = Image.fromarray(merger.merge().astype(np.uint8))
        yield ImageUpscaleResult(
            (np.asarray(ImageOps.flip(step).convert('RGBA'), dtype=np.float32) / 255.) if step is not None else None,
            id + 1,
            tiler.n_tiles,
            (id + 1) == tiler.n_tiles
        )
    if step_preview_mode == StepPreviewMode.NONE:
        final = Image.fromarray(merger.merge().astype(np.uint8))
        yield ImageUpscaleResult(
            np.asarray(ImageOps.flip(final).convert('RGBA'), dtype=np.float32) / 255.,
            tiler.n_tiles,
            tiler.n_tiles,
            True
        )