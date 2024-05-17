from typing import Union, Generator, Callable, List, Optional
import os
from contextlib import nullcontext
import numpy as np
import random
from .prompt_to_image import Checkpoint, Scheduler, Optimizations, StepPreviewMode, step_latents, step_images, _configure_model_padding
from ...api.models.seamless_axes import SeamlessAxes
from ..future import Future
from ...image_utils import image_to_np, size, resize, rgb, ImageOrPath

def inpaint(
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

    iterations: int,

    step_preview_mode: StepPreviewMode,

    inpaint_mask_src: str,
    text_mask: str,
    text_mask_confidence: float,

    # Stability SDK
    key: str | None = None,

    **kwargs
) -> Generator[Future, None, None]:
    future = Future()
    yield future

    import diffusers
    import torch
    
    device = self.choose_device(optimizations)

    # StableDiffusionPipeline w/ caching
    pipe = self.load_model(diffusers.AutoPipelineForInpainting, model, optimizations, scheduler)
    height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor

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
    image = image_to_np(image)
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
        match inpaint_mask_src:
            case 'alpha':
                mask_image = 1-image[..., -1]
                image = rgb(image)
            case 'prompt':
                image = rgb(image)
                from transformers import AutoProcessor, CLIPSegForImageSegmentation

                processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", do_rescale=False)
                clipseg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
                inputs = processor(text=[text_mask], images=[image], return_tensors="pt", padding=True)
                outputs = clipseg(**inputs)
                mask_image = (torch.sigmoid(outputs.logits) >= text_mask_confidence).detach().numpy().astype(np.float32)
                mask_image = resize(mask_image, (width, height))
        
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
                mask_image=[mask_image] * batch_size,
                strength=strength,
                height=height,
                width=width,
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