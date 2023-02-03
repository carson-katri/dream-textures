from typing import Union, Generator, Callable, List, Optional
import os
from contextlib import nullcontext

from numpy.typing import NDArray
import numpy as np
import random
from .prompt_to_image import Scheduler, Optimizations, StepPreviewMode, ImageGenerationResult, _configure_model_padding, model_snapshot_folder, load_pipe
from ..models import Pipeline
from .detect_seamless import SeamlessAxes


def depth_to_image(
    self,
    pipeline: Pipeline,
    
    model: str,

    scheduler: Scheduler,

    optimizations: Optimizations,

    depth: NDArray | None,
    image: NDArray | str | None,
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
) -> Generator[NDArray, None, None]:
    match pipeline:
        case Pipeline.STABLE_DIFFUSION:
            import diffusers
            import torch
            import PIL.Image
            import PIL.ImageOps

            class GeneratorPipeline(diffusers.StableDiffusionInpaintPipeline):
                def prepare_depth(self, depth, image, dtype, device):
                    device = torch.device('cpu' if device.type == 'mps' else device.type)
                    if depth is None:
                        from transformers import DPTFeatureExtractor, DPTForDepthEstimation
                        import contextlib
                        feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
                        depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
                        depth_estimator = depth_estimator.to(device)
                        
                        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
                        pixel_values = pixel_values.to(device=device)
                        # The DPT-Hybrid model uses batch-norm layers which are not compatible with fp16.
                        # So we use `torch.autocast` here for half precision inference.
                        context_manger = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else contextlib.nullcontext()
                        with context_manger:
                            depth_map = depth_estimator(pixel_values).predicted_depth
                        depth_map = torch.nn.functional.interpolate(
                            depth_map.unsqueeze(1),
                            size=(height // self.vae_scale_factor, width // self.vae_scale_factor),
                            mode="bicubic",
                            align_corners=False,
                        )

                        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
                        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
                        depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
                        depth_map = depth_map.to(device)
                        return depth_map
                    else:
                        if isinstance(depth, PIL.Image.Image):
                            depth = np.array(depth.convert("L"))
                            depth = depth.astype(np.float32) / 255.0
                        depth = depth[None, None]
                        depth = torch.from_numpy(depth)
                        return depth
                        
                def prepare_depth_latents(
                    self, depth, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
                ):
                    # resize the mask to latents shape as we concatenate the mask to the latents
                    # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
                    # and half precision
                    depth = torch.nn.functional.interpolate(
                        depth, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
                    )
                    depth = depth.to(device=device, dtype=dtype)

                    # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
                    depth = depth.repeat(batch_size, 1, 1, 1)
                    depth = torch.cat([depth] * 2) if do_classifier_free_guidance else depth
                    return depth

                def prepare_img2img_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None, image=None, timestep=None):
                    shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
                    if isinstance(generator, list) and len(generator) != batch_size:
                        raise ValueError(
                            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                        )

                    if latents is None:
                        rand_device = "cpu" if device.type == "mps" else device

                        if isinstance(generator, list):
                            shape = (1,) + shape[1:]
                            latents = [
                                torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                                for i in range(batch_size)
                            ]
                            latents = torch.cat(latents, dim=0).to(device)
                        else:
                            latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
                    else:
                        if latents.shape != shape:
                            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
                        latents = latents.to(device)

                    # scale the initial noise by the standard deviation required by the scheduler
                    latents = latents * self.scheduler.init_noise_sigma

                    if image is not None:
                        image = image.to(device=device, dtype=dtype)
                        if isinstance(generator, list):
                            image_latents = [
                                self.vae.encode(image[0:1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                            ]
                            image_latents = torch.cat(image_latents, dim=0)
                        else:
                            image_latents = self.vae.encode(image).latent_dist.sample(generator)
                        image_latents = torch.nn.functional.interpolate(
                            image_latents, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
                        )
                        image_latents = 0.18215 * image_latents
                        rand_device = "cpu" if device.type == "mps" else device
                        shape = image_latents.shape
                        if isinstance(generator, list):
                            shape = (1,) + shape[1:]
                            noise = [
                                torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype) for i in
                                range(batch_size)
                            ]
                            noise = torch.cat(noise, dim=0).to(device)
                        else:
                            noise = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
                        latents = self.scheduler.add_noise(image_latents, noise, timestep)

                    return latents


                def get_timesteps(self, num_inference_steps, strength, device):
                    # get the original timestep using init_timestep
                    offset = self.scheduler.config.get("steps_offset", 0)
                    init_timestep = int(num_inference_steps * strength) + offset
                    init_timestep = min(init_timestep, num_inference_steps)

                    t_start = max(num_inference_steps - init_timestep + offset, 0)
                    timesteps = self.scheduler.timesteps[t_start:]

                    return timesteps, num_inference_steps - t_start

                @torch.no_grad()
                def __call__(
                    self,
                    prompt: Union[str, List[str]],
                    depth_image: Union[torch.FloatTensor, PIL.Image.Image],
                    image: Optional[Union[torch.FloatTensor, PIL.Image.Image]] = None,
                    strength: float = 0.8,
                    height: Optional[int] = None,
                    width: Optional[int] = None,
                    num_inference_steps: int = 50,
                    guidance_scale: float = 7.5,
                    negative_prompt: Optional[Union[str, List[str]]] = None,
                    num_images_per_prompt: Optional[int] = 1,
                    eta: float = 0.0,
                    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                    latents: Optional[torch.FloatTensor] = None,
                    output_type: Optional[str] = "pil",
                    return_dict: bool = True,
                    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
                    callback_steps: Optional[int] = 1,
                    **kwargs,
                ):
                    
                    # 0. Default height and width to unet
                    height = height or self.unet.config.sample_size * self.vae_scale_factor
                    width = width or self.unet.config.sample_size * self.vae_scale_factor

                    # 1. Check inputs
                    self.check_inputs(prompt, height, width, callback_steps)

                    # 2. Define call parameters
                    batch_size = 1 if isinstance(prompt, str) else len(prompt)
                    device = self._execution_device
                    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
                    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
                    # corresponds to doing no classifier free guidance.
                    do_classifier_free_guidance = guidance_scale > 1.0

                    # 3. Encode input prompt
                    text_embeddings = self._encode_prompt(
                        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
                    )

                    # 4. Prepare the depth image
                    depth = self.prepare_depth(depth_image, image, text_embeddings.dtype, device)

                    if image is not None and isinstance(image, PIL.Image.Image):
                        image = diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess(image)

                    # 5. set timesteps
                    self.scheduler.set_timesteps(num_inference_steps, device=device)
                    timesteps = self.scheduler.timesteps
                    if image is not None:
                        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

                    # 6. Prepare latent variables
                    num_channels_latents = self.vae.config.latent_channels
                    if image is not None:
                        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
                        latents = self.prepare_img2img_latents(
                            batch_size * num_images_per_prompt,
                            num_channels_latents,
                            height,
                            width,
                            text_embeddings.dtype,
                            device,
                            generator,
                            latents,
                            image,
                            latent_timestep
                        )
                    else:
                        latents = self.prepare_latents(
                            batch_size * num_images_per_prompt,
                            num_channels_latents,
                            height,
                            width,
                            text_embeddings.dtype,
                            device,
                            generator,
                            latents,
                        )

                    # 7. Prepare mask latent variables
                    depth = self.prepare_depth_latents(
                        depth,
                        batch_size * num_images_per_prompt,
                        height,
                        width,
                        text_embeddings.dtype,
                        device,
                        generator,
                        do_classifier_free_guidance,
                    )

                    # 8. Check that sizes of mask, masked image and latents match
                    num_channels_depth = depth.shape[1]
                    if num_channels_latents + num_channels_depth != self.unet.config.in_channels:
                        raise ValueError(
                            f"Select a depth model, such as 'stabilityai/stable-diffusion-2-depth'"
                        )

                    # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
                    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

                    # 10. Denoising loop
                    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
                    with self.progress_bar(total=num_inference_steps) as progress_bar:
                        for i, t in enumerate(timesteps):
                            # expand the latents if we are doing classifier free guidance
                            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                            # concat latents, mask, masked_image_latents in the channel dimension
                            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                            latent_model_input = torch.cat([latent_model_input, depth], dim=1)

                            # predict the noise residual
                            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                            # perform guidance
                            if do_classifier_free_guidance:
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                            # compute the previous noisy sample x_t -> x_t-1
                            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                                progress_bar.update()
                            # NOTE: Modified to yield the latents instead of calling a callback.
                            yield ImageGenerationResult.step_preview(self, kwargs['step_preview_mode'], width, height, latents, generator, i)

                    # 11. Post-processing
                    image = self.decode_latents(latents)

                    # TODO: Add UI to enable this.
                    # 12. Run safety checker
                    # image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

                    # NOTE: Modified to yield the decoded image as a numpy array.
                    yield ImageGenerationResult(
                        [np.asarray(PIL.ImageOps.flip(image).convert('RGBA'), dtype=np.float32) / 255.
                            for i, image in enumerate(self.numpy_to_pil(image))],
                        [gen.initial_seed() for gen in generator] if isinstance(generator, list) else [generator.initial_seed()],
                        num_inference_steps,
                        True
                    )
            
            if optimizations.cpu_only:
                device = "cpu"
            else:
                device = self.choose_device()

            # StableDiffusionPipeline w/ caching
            pipe = load_pipe(self, "depth", GeneratorPipeline, model, optimizations, scheduler, device)

            # Optimizations
            pipe = optimizations.apply(pipe, device)

            # RNG
            batch_size = len(prompt) if isinstance(prompt, list) else 1
            generator = []
            for _ in range(batch_size):
                gen = torch.Generator(device="cpu" if device in ("mps", "privateuseone") else device) # MPS and DML do not support the `Generator` API
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
            depth_image = PIL.ImageOps.flip(PIL.Image.fromarray(np.uint8(depth * 255)).convert('L')).resize(rounded_size) if depth is not None else None
            init_image = None if image is None else (PIL.Image.open(image) if isinstance(image, str) else PIL.Image.fromarray(image.astype(np.uint8))).convert('RGB').resize(rounded_size)

            # Seamless
            if seamless_axes == SeamlessAxes.AUTO:
                init_sa = None if init_image is None else self.detect_seamless(np.array(init_image) / 255)
                depth_sa = None if depth_image is None else self.detect_seamless(np.array(depth_image.convert('RGB')) / 255)
                if init_sa is not None and depth_sa is not None:
                    seamless_axes = SeamlessAxes((init_sa.x and depth_sa.x, init_sa.y and depth_sa.y))
                elif init_sa is not None:
                    seamless_axes = init_sa
                elif depth_sa is not None:
                    seamless_axes = depth_sa
            _configure_model_padding(pipe.unet, seamless_axes)
            _configure_model_padding(pipe.vae, seamless_axes)

            # Inference
            with (torch.inference_mode() if device not in ('mps', "privateuseone") else nullcontext()), \
                (torch.autocast(device) if optimizations.can_use("amp", device) else nullcontext()):
                yield from pipe(
                    prompt=prompt,
                    depth_image=depth_image,
                    image=init_image,
                    strength=strength,
                    width=rounded_size[0],
                    height=rounded_size[1],
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    negative_prompt=negative_prompt if use_negative_prompt else None,
                    num_images_per_prompt=1,
                    eta=0.0,
                    generator=generator,
                    latents=None,
                    output_type="pil",
                    return_dict=True,
                    callback=None,
                    callback_steps=1,
                    step_preview_mode=step_preview_mode
                )
        case Pipeline.STABILITY_SDK:
            import stability_sdk
            raise NotImplementedError()
        case _:
            raise Exception(f"Unsupported pipeline {pipeline}.")