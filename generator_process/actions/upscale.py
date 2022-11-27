import numpy as np
from .prompt_to_image import Optimizations, Scheduler

def upscale(
    self,
    image: str,
    
    prompt: str,
    steps: int,
    seed: int,
    cfg_scale: float,
    scheduler: Scheduler,
    
    tile_size: int,

    optimizations: Optimizations,

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
        revision="fp16" if optimizations.can_use("half_precision", device) else None,
        torch_dtype=torch.float16 if optimizations.can_use("half_precision", device) else torch.float32
    )
    pipe.scheduler = scheduler.create(pipe, None)
    pipe = pipe.to(device)
    pipe = optimizations.apply(pipe, device)

    low_res_image = Image.open(image)

    generator = torch.Generator() if seed is None else (torch.manual_seed(seed) if device == "mps" else torch.Generator(device=device).manual_seed(seed))
    initial_seed = generator.initial_seed()

    final = Image.new('RGB', (low_res_image.size[0] * 4, low_res_image.size[1] * 4))
    for x in range(low_res_image.size[0] // tile_size):
        for y in range(low_res_image.size[1] // tile_size):
            x_offset = x * tile_size
            y_offset = y * tile_size
            tile = low_res_image.crop((x_offset, y_offset, x_offset + tile_size, y_offset + tile_size))
            upscaled = pipe(
                prompt=prompt,
                image=tile,
                num_inference_steps=steps,
                generator=torch.manual_seed(initial_seed),
                guidance_scale=cfg_scale,

            ).images[0]
            final.paste(upscaled, (x_offset * 4, y_offset * 4))
            yield np.asarray(ImageOps.flip(final).convert('RGBA'), dtype=np.float32) / 255.

    yield np.asarray(ImageOps.flip(final).convert('RGBA'), dtype=np.float32) / 255.