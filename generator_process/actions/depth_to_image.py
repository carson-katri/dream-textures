from typing import Union, Generator, Callable, List, Optional
import os
from contextlib import nullcontext
from numpy.typing import NDArray
import numpy as np
from .prompt_to_image import Pipeline, Scheduler, Optimizations, StepPreviewMode, approximate_decoded_latents, _configure_model_padding

def depth_to_image(
    self,
    pipeline: Pipeline,
    
    model: str,

    scheduler: Scheduler,

    optimizations: Optimizations,

    depth: NDArray | str,
    prompt: str,
    steps: int,
    seed: int,

    cfg_scale: float,
    use_negative_prompt: bool,
    negative_prompt: str,

    step_preview_mode: StepPreviewMode,

    **kwargs
) -> Generator[NDArray, None, None]:
    match pipeline:
        case Pipeline.STABLE_DIFFUSION:
            import diffusers
            import torch
            import PIL.Image
            import PIL.ImageOps
            from ...absolute_path import WEIGHTS_PATH

            final_size = depth.shape[:2]

            def prepare_depth(depth):
                if isinstance(depth, PIL.Image.Image):
                    depth = np.array(depth.convert("L"))
                    depth = depth.astype(np.float32) / 255.0
                depth = depth[None, None]
                depth[depth < 0.5] = 0
                depth[depth >= 0.5] = 1
                depth = torch.from_numpy(depth)
                return depth

            class GeneratorPipeline(diffusers.StableDiffusionInpaintPipeline):
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

                def prepare_img2img_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
                    image = image.to(device=device, dtype=dtype)
                    init_latent_dist = self.vae.encode(image).latent_dist
                    init_latents = init_latent_dist.sample(generator=generator)
                    init_latents = 0.18215 * init_latents

                    if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
                        # expand init_latents for batch_size
                        deprecation_message = (
                            f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                            " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                            " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                            " your script to pass as many initial images as text prompts to suppress this warning."
                        )
                        deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
                        additional_image_per_prompt = batch_size // init_latents.shape[0]
                        init_latents = torch.cat([init_latents] * additional_image_per_prompt * num_images_per_prompt, dim=0)
                    elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
                        raise ValueError(
                            f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
                        )
                    else:
                        init_latents = torch.cat([init_latents] * num_images_per_prompt, dim=0)

                    # add noise to latents using the timesteps
                    noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=dtype)

                    # get latents
                    init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
                    latents = init_latents

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
                    generator: Optional[torch.Generator] = None,
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
                    depth = prepare_depth(depth_image)

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
                            image, latent_timestep, batch_size, num_images_per_prompt, text_embeddings.dtype, device, generator
                        )
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
                            f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                            f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                            f" `num_channels_mask`: {num_channels_depth}"
                            f" = {num_channels_latents+num_channels_depth}. Please verify the config of"
                            " `pipeline.unet` or your `mask_image` or `image` input."
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

                            # call the callback, if provided
                            match kwargs['step_preview_mode']:
                                case StepPreviewMode.NONE:
                                    pass
                                case StepPreviewMode.FAST:
                                    yield np.asarray(PIL.ImageOps.flip(PIL.Image.fromarray(approximate_decoded_latents(latents))).resize(final_size, PIL.Image.Resampling.NEAREST).convert('RGBA'), dtype=np.float32) / 255.
                                case StepPreviewMode.ACCURATE:
                                    yield from [
                                        np.asarray(PIL.ImageOps.flip(image).resize(final_size).convert('RGBA'), dtype=np.float32) / 255.
                                        for image in self.numpy_to_pil(self.decode_latents(latents))
                                    ]

                    # 11. Post-processing
                    image = self.decode_latents(latents)

                    # TODO: Add UI to enable this.
                    # 12. Run safety checker
                    # image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

                    # NOTE: Modified to yield the decoded image as a numpy array.
                    yield from [
                        np.asarray(PIL.ImageOps.flip(image).resize(final_size).convert('RGBA'), dtype=np.float32) / 255.
                        for image in self.numpy_to_pil(image)
                    ]
            
            if optimizations.cpu_only:
                device = "cpu"
            else:
                device = self.choose_device()

            use_cpu_offload = optimizations.can_use("sequential_cpu_offload", device)

            # StableDiffusionPipeline w/ caching
            if hasattr(self, "_cached_depth2img_pipe") and self._cached_depth2img_pipe[1] == model and use_cpu_offload == self._cached_depth2img_pipe[2]:
                pipe = self._cached_depth2img_pipe[0]
            else:
                storage_folder = os.path.join(WEIGHTS_PATH, model)
                revision = "main"
                ref_path = os.path.join(storage_folder, "refs", revision)
                with open(ref_path) as f:
                    commit_hash = f.read()

                snapshot_folder = os.path.join(storage_folder, "snapshots", commit_hash)
                pipe = GeneratorPipeline.from_pretrained(
                    "carsonkatri/stable-diffusion-2-depth-diffusers",
                    revision="fp16" if optimizations.can_use("half_precision", device) else None,
                    torch_dtype=torch.float16 if optimizations.can_use("half_precision", device) else torch.float32,
                )
                pipe = pipe.to(device)
                setattr(self, "_cached_depth2img_pipe", (pipe, model, use_cpu_offload, snapshot_folder))

            # Scheduler
            is_stable_diffusion_2 = 'stabilityai--stable-diffusion-2' in model
            pipe.scheduler = scheduler.create(pipe, {
                'model_path': self._cached_depth2img_pipe[3],
                'subfolder': 'scheduler',
            } if is_stable_diffusion_2 else None)

            # Optimizations
            pipe = optimizations.apply(pipe, device)

            # RNG
            generator = None if seed is None else (torch.manual_seed(seed) if device == "mps" else torch.Generator(device=device).manual_seed(seed))

            # Inference
            rounded_size = (
                int(8 * (depth.shape[1] // 8)),
                int(8 * (depth.shape[0] // 8)),
            )
            with (torch.inference_mode() if optimizations.can_use("inference_mode", device) else nullcontext()), \
                    (torch.autocast(device) if optimizations.can_use("amp", device) else nullcontext()):
                    PIL.ImageOps.flip(PIL.Image.fromarray(np.uint8(depth * 255), 'L')).resize(rounded_size).save('test.png')
                    yield from pipe(
                        prompt=prompt,
                        depth_image=PIL.ImageOps.flip(PIL.Image.fromarray(np.uint8(depth * 255), 'L')).resize(rounded_size),
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