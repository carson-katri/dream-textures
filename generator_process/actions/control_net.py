from typing import Union, Generator, Callable, List, Optional, Dict, Any
from contextlib import nullcontext

from numpy.typing import NDArray
import numpy as np
import random
from .prompt_to_image import Scheduler, Optimizations, StepPreviewMode, ImageGenerationResult, _configure_model_padding, model_snapshot_folder, load_pipe
from ..models import Pipeline
from .detect_seamless import SeamlessAxes

def control_net(
    self,
    pipeline: Pipeline,
    
    model: str,

    scheduler: Scheduler,

    optimizations: Optimizations,

    control_net: list[str],
    control: list[NDArray] | None,
    controlnet_conditioning_scale: list[float],
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
            from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet import MultiControlNetModel, ControlNetModel
            import torch
            import PIL.Image
            import PIL.ImageOps

            class GeneratorPipeline(diffusers.StableDiffusionControlNetPipeline):
                @torch.no_grad()
                def __call__(
                    self,
                    prompt: Union[str, List[str]] = None,
                    image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]] = None,
                    height: Optional[int] = None,
                    width: Optional[int] = None,
                    num_inference_steps: int = 50,
                    guidance_scale: float = 7.5,
                    negative_prompt: Optional[Union[str, List[str]]] = None,
                    num_images_per_prompt: Optional[int] = 1,
                    eta: float = 0.0,
                    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                    latents: Optional[torch.FloatTensor] = None,
                    prompt_embeds: Optional[torch.FloatTensor] = None,
                    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
                    output_type: Optional[str] = "pil",
                    return_dict: bool = True,
                    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
                    callback_steps: int = 1,
                    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                    controlnet_conditioning_scale: Union[float, List[float]] = 1.0,

                    **kwargs
                ):
                    # 0. Default height and width to unet
                    height, width = self._default_height_width(height, width, image)

                    # 1. Check inputs. Raise error if not correct
                    self.check_inputs(
                        prompt,
                        image,
                        height,
                        width,
                        callback_steps,
                        negative_prompt,
                        prompt_embeds,
                        negative_prompt_embeds,
                        controlnet_conditioning_scale
                    )

                    # 2. Define call parameters
                    if prompt is not None and isinstance(prompt, str):
                        batch_size = 1
                    elif prompt is not None and isinstance(prompt, list):
                        batch_size = len(prompt)
                    else:
                        batch_size = prompt_embeds.shape[0]

                    device = self._execution_device
                    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
                    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
                    # corresponds to doing no classifier free guidance.
                    do_classifier_free_guidance = guidance_scale > 1.0

                    if isinstance(self.controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
                        controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(self.controlnet.nets)

                    # 3. Encode input prompt
                    prompt_embeds = self._encode_prompt(
                        prompt,
                        device,
                        num_images_per_prompt,
                        do_classifier_free_guidance,
                        negative_prompt,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                    )

                    # 4. Prepare image
                    if isinstance(self.controlnet, ControlNetModel):
                        image = self.prepare_image(
                            image=image,
                            width=width,
                            height=height,
                            batch_size=batch_size * num_images_per_prompt,
                            num_images_per_prompt=num_images_per_prompt,
                            device=device,
                            dtype=self.controlnet.dtype,
                            do_classifier_free_guidance=do_classifier_free_guidance,
                        )
                    elif isinstance(self.controlnet, MultiControlNetModel):
                        images = []

                        for image_ in image:
                            image_ = self.prepare_image(
                                image=image_,
                                width=width,
                                height=height,
                                batch_size=batch_size * num_images_per_prompt,
                                num_images_per_prompt=num_images_per_prompt,
                                device=device,
                                dtype=self.controlnet.dtype,
                                do_classifier_free_guidance=do_classifier_free_guidance,
                            )

                            images.append(image_)

                        image = images
                    else:
                        assert False

                    # 5. Prepare timesteps
                    self.scheduler.set_timesteps(num_inference_steps, device=device)
                    timesteps = self.scheduler.timesteps

                    # 6. Prepare latent variables
                    num_channels_latents = self.unet.in_channels
                    latents = self.prepare_latents(
                        batch_size * num_images_per_prompt,
                        num_channels_latents,
                        height,
                        width,
                        prompt_embeds.dtype,
                        device,
                        generator,
                        latents,
                    )

                    # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
                    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

                    # 8. Denoising loop
                    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
                    with self.progress_bar(total=num_inference_steps) as progress_bar:
                        for i, t in enumerate(timesteps):
                            # expand the latents if we are doing classifier free guidance
                            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                            # controlnet(s) inference
                            down_block_res_samples, mid_block_res_sample = self.controlnet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=prompt_embeds,
                                controlnet_cond=image,
                                conditioning_scale=controlnet_conditioning_scale,
                                return_dict=False,
                            )

                            # predict the noise residual
                            noise_pred = self.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=prompt_embeds,
                                cross_attention_kwargs=cross_attention_kwargs,
                                down_block_additional_residuals=down_block_res_samples,
                                mid_block_additional_residual=mid_block_res_sample,
                            ).sample

                            # perform guidance
                            if do_classifier_free_guidance:
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                            # compute the previous noisy sample x_t -> x_t-1
                            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                            # NOTE: Modified to yield the latents instead of calling a callback.
                            yield ImageGenerationResult.step_preview(self, kwargs['step_preview_mode'], width, height, latents, generator, i)

                    # If we do sequential model offloading, let's offload unet and controlnet
                    # manually for max memory savings
                    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                        self.unet.to("cpu")
                        self.controlnet.to("cpu")
                        torch.cuda.empty_cache()

                    if output_type == "latent":
                        image = latents
                        has_nsfw_concept = None
                    elif output_type == "pil":
                        # 8. Post-processing
                        image = self.decode_latents(latents)

                        # NOTE: Add UI to enable this.
                        # 9. Run safety checker
                        # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

                        # 10. Convert to PIL
                        image = self.numpy_to_pil(image)
                    else:
                        # 8. Post-processing
                        image = self.decode_latents(latents)

                        # NOTE: Add UI to enable this.
                        # 9. Run safety checker
                        # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

                    # Offload last model to CPU
                    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                        self.final_offload_hook.offload()

                    # NOTE: Modified to yield the decoded image as a numpy array.
                    yield ImageGenerationResult(
                        [np.asarray(PIL.ImageOps.flip(image).convert('RGBA'), dtype=np.float32) / 255.
                            for i, image in enumerate(image)],
                        [gen.initial_seed() for gen in generator] if isinstance(generator, list) else [generator.initial_seed()],
                        num_inference_steps,
                        True
                    )
            
            if optimizations.cpu_only:
                device = "cpu"
            else:
                device = self.choose_device()

            # Load the ControlNet model
            controlnet = []
            for controlnet_name in control_net:
                controlnet.append(load_pipe(self, f"control_net_model-{controlnet_name}", diffusers.ControlNetModel, controlnet_name, optimizations, None, device))
            controlnet = MultiControlNetModel(controlnet)

            # StableDiffusionPipeline w/ caching
            pipe = load_pipe(self, "control_net", GeneratorPipeline, model, optimizations, scheduler, device, controlnet=controlnet)

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
            print(control)
            control_image = [PIL.Image.fromarray(np.uint8(c * 255)).convert('RGB').resize(rounded_size) for c in control] if control is not None else None
            init_image = None if image is None else (PIL.Image.open(image) if isinstance(image, str) else PIL.Image.fromarray(image.astype(np.uint8))).convert('RGB').resize(rounded_size)

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
            with (torch.inference_mode() if device not in ('mps', "privateuseone") else nullcontext()), \
                (torch.autocast(device) if optimizations.can_use("amp", device) else nullcontext()):
                yield from pipe(
                    prompt=prompt,
                    image=control_image,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    # image=init_image,
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