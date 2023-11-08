from typing import Generator
from contextlib import nullcontext

import numpy as np
import random
from ...api.models.seamless_axes import SeamlessAxes
from ...api.models.step_preview_mode import StepPreviewMode
from ..models import Checkpoint, Optimizations, Scheduler
from ..models.image_generation_result import ImageGenerationResult
from ..future import Future

def prompt_to_image(
    self,
    
    model: str | Checkpoint,

    scheduler: str | Scheduler,

    optimizations: Optimizations,

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

    # Stability SDK
    key: str | None = None,

    sdxl_refiner_model: str | Checkpoint | None = None,

    **kwargs
) -> Generator[Future, None, None]:
    future = Future()
    yield future

    import diffusers
    import torch
    from PIL import Image, ImageOps
    from diffusers.pipelines.wuerstchen import WuerstchenCombinedPipeline

    device = self.choose_device(optimizations)

    # Stable Diffusion pipeline w/ caching
    if sdxl_refiner_model is not None and device == "cuda" and (optimizations.cpu_offloading(device) or torch.cuda.mem_get_info()[1] > 20 * 1024**3 * (1 if optimizations.can_use_half(device) else 2)):
        pipe, refiner = self.load_model(diffusers.AutoPipelineForText2Image, model, optimizations, scheduler, sdxl_refiner_model=sdxl_refiner_model)
    else:
        pipe = self.load_model(diffusers.AutoPipelineForText2Image, model, optimizations, scheduler)
        refiner = None
    height = height or ((pipe.unet.config.sample_size * pipe.vae_scale_factor) if hasattr(pipe, 'unet') and hasattr(pipe, 'vae_scale_factor') else 512)
    width = width or ((pipe.unet.config.sample_size * pipe.vae_scale_factor) if hasattr(pipe, 'unet') and hasattr(pipe, 'vae_scale_factor') else 512)

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
    if hasattr(pipe, 'unet'):
        _configure_model_padding(pipe.unet, seamless_axes)
    if hasattr(pipe, 'vae'):
        _configure_model_padding(pipe.vae, seamless_axes)
    if hasattr(pipe, 'prior_prior'):
        _configure_model_padding(pipe.prior_prior, seamless_axes)
    if hasattr(pipe, 'decoder'):
        _configure_model_padding(pipe.decoder, seamless_axes)
    if hasattr(pipe, 'vqgan'):
        _configure_model_padding(pipe.vqgan, seamless_axes)

    # Inference
    with torch.inference_mode() if device not in ('mps', "dml") else nullcontext():
        is_sdxl = isinstance(pipe, diffusers.StableDiffusionXLPipeline)
        output_type = "latent" if is_sdxl and sdxl_refiner_model is not None else "pil"
        def callback(step, timestep, latents):
            if future.check_cancelled():
                raise InterruptedError()
            future.add_response(ImageGenerationResult.step_preview(self, step_preview_mode, width, height, latents, generator, step))
        try:
            pipe_kwargs = {
                'prompt': prompt,
                'height': height,
                'width': width,
                'num_inference_steps': steps,
                'guidance_scale': cfg_scale,
                'negative_prompt': negative_prompt if use_negative_prompt else None,
                'num_images_per_prompt': 1,
                'eta': 0.0,
                'generator': generator,
                'latents': None,
                'output_type': output_type,
                'return_dict': True,
                'callback': callback,
                'callback_steps': 1,
            }
            if isinstance(pipe, WuerstchenCombinedPipeline):
                pipe_kwargs['prior_guidance_scale'] = pipe_kwargs.pop('guidance_scale')
                del pipe_kwargs['eta']
            result = pipe(**pipe_kwargs)
            if is_sdxl and sdxl_refiner_model is not None and refiner is None:
                # allow load_model() to garbage collect pipe
                pipe = None
                refiner = self.load_model(diffusers.AutoPipelineForImage2Image, sdxl_refiner_model, optimizations, scheduler)
            if refiner is not None:
                refiner = optimizations.apply(refiner, device)
                result = refiner(
                    prompt=prompt,
                    negative_prompt=[""],
                    callback=callback,
                    callback_steps=1,
                    num_inference_steps=steps,
                    image=result.images
                )
            
            future.add_response(ImageGenerationResult(
                [np.asarray(ImageOps.flip(image).convert('RGBA'), dtype=np.float32) / 255.
                    for image in result.images],
                [gen.initial_seed() for gen in generator] if isinstance(generator, list) else [generator.initial_seed()],
                steps,
                True
            ))
        except InterruptedError:
            pass
    
    future.set_done()

def _conv_forward_asymmetric(self, input, weight, bias):
    import torch.nn as nn
    """
    Patch for Conv2d._conv_forward that supports asymmetric padding
    """
    if input.device.type == "dml":
        # DML pad() will wrongly fill the tensor in constant mode with the supplied value
        # (default 0) when padding on both ends of a dimension, can't split to two calls.
        working = nn.functional.pad(input, self._reversed_padding_repeated_twice, mode='circular')
        pad_w0, pad_w1, pad_h0, pad_h1 = self._reversed_padding_repeated_twice
        if self.asymmetric_padding_mode[0] == 'constant':
            working[:, :, :, :pad_w0] = 0
            if pad_w1 > 0:
                working[:, :, :, -pad_w1:] = 0
        if self.asymmetric_padding_mode[1] == 'constant':
            working[:, :, :pad_h0] = 0
            if pad_h1 > 0:
                working[:, :, -pad_h1:] = 0
    else:
        working = nn.functional.pad(input, self.asymmetric_padding[0], mode=self.asymmetric_padding_mode[0])
        working = nn.functional.pad(working, self.asymmetric_padding[1], mode=self.asymmetric_padding_mode[1])
    return nn.functional.conv2d(working, weight, bias, self.stride, nn.modules.utils._pair(0), self.dilation, self.groups)

def _lora_compatible_conv_forward(self, hidden_states, scale=1.0):
    return self._conv_forward(hidden_states, self.weight, self.bias)

def _configure_model_padding(model, seamless_axes):
    import torch.nn as nn
    from diffusers.models.lora import LoRACompatibleConv
    """
    Modifies the 2D convolution layers to use a circular padding mode based on the `seamless` and `seamless_axes` options.
    """
    seamless_axes = SeamlessAxes(seamless_axes)
    if seamless_axes == SeamlessAxes.AUTO:
        seamless_axes = seamless_axes.OFF
    if getattr(model, "seamless_axes", SeamlessAxes.OFF) == seamless_axes:
        return
    model.seamless_axes = seamless_axes
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, LoRACompatibleConv)):
            if seamless_axes.x or seamless_axes.y:
                if isinstance(m, LoRACompatibleConv):
                    m.forward = _lora_compatible_conv_forward.__get__(m, LoRACompatibleConv)
                m.asymmetric_padding_mode = (
                    'circular' if seamless_axes.x else 'constant',
                    'circular' if seamless_axes.y else 'constant'
                )
                m.asymmetric_padding = (
                    (m._reversed_padding_repeated_twice[0], m._reversed_padding_repeated_twice[1], 0, 0),
                    (0, 0, m._reversed_padding_repeated_twice[2], m._reversed_padding_repeated_twice[3])
                )
                m._conv_forward = _conv_forward_asymmetric.__get__(m, nn.Conv2d)
            else:
                if isinstance(m, LoRACompatibleConv):
                    m.forward = LoRACompatibleConv.forward.__get__(m, LoRACompatibleConv)
                m._conv_forward = nn.Conv2d._conv_forward.__get__(m, nn.Conv2d)
                if hasattr(m, 'asymmetric_padding_mode'):
                    del m.asymmetric_padding_mode
                if hasattr(m, 'asymmetric_padding'):
                    del m.asymmetric_padding
