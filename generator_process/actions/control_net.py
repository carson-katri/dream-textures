from typing import Union, Generator, Callable, List, Optional, Dict, Any
from contextlib import nullcontext

from numpy.typing import NDArray
import numpy as np
import random
from .prompt_to_image import Scheduler, Optimizations, StepPreviewMode, ImageGenerationResult, _configure_model_padding
from ...api.models.seamless_axes import SeamlessAxes
from ..future import Future

def control_net(
    self,

    model: str,

    scheduler: Scheduler,

    optimizations: Optimizations,

    control_net: list[str],
    control: list[NDArray] | None,
    controlnet_conditioning_scale: list[float],
    
    image: NDArray | str | None, # image to image
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
    from diffusers.models.controlnet import ControlNetModel
    from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
    from diffusers.utils import deprecate, randn_tensor
    import torch
    import PIL.Image
    import PIL.ImageOps
    
    if optimizations.cpu_only:
        device = "cpu"
    else:
        device = self.choose_device(optimizations)

    # Load the ControlNet model
    controlnet_models = MultiControlNetModel([
        ControlNetModel.from_pretrained(name) for name in control_net
    ])

    # StableDiffusionPipeline w/ caching
    if image is not None:
        if inpaint:
            pipe = self.load_model(diffusers.AutoPipelineForInpainting, model, controlnet=controlnet_models)
        else:
            pipe = self.load_model(diffusers.AutoPipelineForImage2Image, model, controlnet=controlnet_models)
    else:
        pipe = self.load_model(diffusers.AutoPipelineForText2Image, model, controlnet=controlnet_models)

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
    control_image = [PIL.Image.fromarray(np.uint8(c * 255)).convert('RGB').resize(rounded_size) for c in control] if control is not None else None
    init_image = None if image is None else (PIL.Image.open(image) if isinstance(image, str) else PIL.Image.fromarray(image.astype(np.uint8))).resize(rounded_size)
    if inpaint:
        match inpaint_mask_src:
            case 'alpha':
                mask_image = PIL.ImageOps.invert(init_image.getchannel('A'))
            case 'prompt':
                from transformers import AutoProcessor, CLIPSegForImageSegmentation

                processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
                clipseg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
                inputs = processor(text=[text_mask], images=[init_image.convert('RGB')], return_tensors="pt", padding=True)
                outputs = clipseg(**inputs)
                mask_image = PIL.Image.fromarray(np.uint8((1 - torch.sigmoid(outputs.logits).lt(text_mask_confidence).int().detach().numpy()) * 255), 'L').resize(init_image.size)
    else:
        mask_image = None

    # Seamless
    if seamless_axes == SeamlessAxes.AUTO:
        init_sa = None if init_image is None else self.detect_seamless(np.array(init_image) / 255)
        control_sa = None if control_image is None else self.detect_seamless(np.array(control_image[0]) / 255)
        if init_sa is not None and control_sa is not None:
            seamless_axes = SeamlessAxes((init_sa.x and control_sa.x, init_sa.y and control_sa.y))
        elif init_sa is not None:
            seamless_axes = init_sa
        elif control_sa is not None:
            seamless_axes = control_sa
    _configure_model_padding(pipe.unet, seamless_axes)
    _configure_model_padding(pipe.vae, seamless_axes)

    # Inference
    with (torch.inference_mode() if device not in ('mps', "dml") else nullcontext()), \
        (torch.autocast(device) if optimizations.can_use("amp", device) else nullcontext()):
        def callback(step, timestep, latents):
            future.add_response(ImageGenerationResult.step_preview(self, step_preview_mode, width, height, latents, generator, step))
        if init_image is not None:
            if mask_image is not None:
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if use_negative_prompt else None,
                    control_image=control_image,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    image=init_image.convert('RGB'),
                    mask_image=mask_image,
                    strength=strength,
                    width=rounded_size[0],
                    height=rounded_size[1],
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    generator=generator,
                    callback=callback
                )
            else:
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if use_negative_prompt else None,
                    control_image=control_image,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    image=init_image.convert('RGB'),
                    strength=strength,
                    width=rounded_size[0],
                    height=rounded_size[1],
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    generator=generator,
                    callback=callback
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
                callback=callback
            )

        future.add_response(ImageGenerationResult(
            [np.asarray(PIL.ImageOps.flip(image).convert('RGBA'), dtype=np.float32) / 255.
                for image in result.images],
            [gen.initial_seed() for gen in generator] if isinstance(generator, list) else [generator.initial_seed()],
            steps,
            True
        ))
    
    future.set_done()