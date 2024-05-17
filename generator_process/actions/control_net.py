from typing import Union, Generator, Callable, List, Optional, Dict, Any
from contextlib import nullcontext

import numpy as np
import logging
import os
import random
from .prompt_to_image import Checkpoint, Scheduler, Optimizations, StepPreviewMode, step_latents, step_images, _configure_model_padding
from ...api.models.seamless_axes import SeamlessAxes
from ..future import Future
from ...image_utils import image_to_np, rgb, resize, ImageOrPath


def control_net(
    self,

    model: str | Checkpoint,

    scheduler: str | Scheduler,

    optimizations: Optimizations,

    control_net: list[str | Checkpoint],
    control: list[ImageOrPath] | None,
    controlnet_conditioning_scale: list[float],
    
    image: ImageOrPath | None, # image to image
    # inpaint
    inpaint: bool,
    inpaint_mask_src: str,
    text_mask: str,
    text_mask_confidence: float,

    strength: float,
    prompt: str | list[str],
    steps: int,
    seed: int,

    width: int | None,
    height: int | None,

    cfg_scale: float,
    use_negative_prompt: bool,
    negative_prompt: str,

    seamless_axes: SeamlessAxes | str | bool | tuple[bool, bool] | None,

    step_preview_mode: StepPreviewMode,

    **kwargs
) -> Generator[Future, None, None]:
    future = Future()
    yield future

    import diffusers
    import torch
    
    device = self.choose_device(optimizations)

    # StableDiffusionPipeline w/ caching
    if image is not None:
        if inpaint:
            pipe = self.load_model(diffusers.AutoPipelineForInpainting, model, optimizations, scheduler, controlnet=control_net)
        else:
            pipe = self.load_model(diffusers.AutoPipelineForImage2Image, model, optimizations, scheduler, controlnet=control_net)
    else:
        pipe = self.load_model(diffusers.AutoPipelineForText2Image, model, optimizations, scheduler, controlnet=control_net)

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
    # FIXME: The `unet.config.sample_size` of the depth model is `32`, not `64`. For now, this will be hardcoded to `512`.
    height = height or 512
    width = width or 512
    rounded_size = (
        int(8 * (width // 8)),
        int(8 * (height // 8)),
    )
    # StableDiffusionControlNetPipeline.check_image() currently fails without adding batch dimension
    control_image = None if control is None else [image_to_np(c, mode="RGB", size=rounded_size)[np.newaxis] for c in control]
    image = image_to_np(image, size=rounded_size)
    if inpaint:
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
    else:
        mask_image = None

    # Seamless
    if seamless_axes == SeamlessAxes.AUTO:
        init_sa = None if image is None else self.detect_seamless(image)
        control_sa = None if control_image is None else self.detect_seamless(control_image[0][0])
        if init_sa is not None and control_sa is not None:
            seamless_axes = init_sa & control_sa
        elif init_sa is not None:
            seamless_axes = init_sa
        elif control_sa is not None:
            seamless_axes = control_sa
    _configure_model_padding(pipe.unet, seamless_axes)
    _configure_model_padding(pipe.vae, seamless_axes)

    # Inference
    with (torch.inference_mode() if device not in ('mps', "dml") else nullcontext()), \
        (torch.autocast(device) if optimizations.can_use("amp", device) else nullcontext()):
        def callback(pipe, step, timestep, callback_kwargs):
            if future.check_cancelled():
                raise InterruptedError()
            future.add_response(step_latents(pipe, step_preview_mode, callback_kwargs["latents"], generator, step, steps))
            return callback_kwargs
        try:
            if image is not None:
                if mask_image is not None:
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt if use_negative_prompt else None,
                        control_image=control_image,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        image=image,
                        mask_image=mask_image,
                        strength=strength,
                        width=rounded_size[0],
                        height=rounded_size[1],
                        num_inference_steps=steps,
                        guidance_scale=cfg_scale,
                        generator=generator,
                        callback_on_step_end=callback,
                        callback_steps=1,
                        output_type="np"
                    )
                else:
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt if use_negative_prompt else None,
                        control_image=control_image,
                        controlnet_conditioning_scale=controlnet_conditioning_scale,
                        image=image,
                        strength=strength,
                        width=rounded_size[0],
                        height=rounded_size[1],
                        num_inference_steps=steps,
                        guidance_scale=cfg_scale,
                        generator=generator,
                        callback_on_step_end=callback,
                        callback_steps=1,
                        output_type="np"
                    )
            else:
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if use_negative_prompt else None,
                    image=control_image,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    width=rounded_size[0],
                    height=rounded_size[1],
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