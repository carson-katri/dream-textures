from typing import Union, Generator, Callable, List, Optional
import os
from contextlib import nullcontext

from numpy.typing import NDArray
import numpy as np
import random
from .prompt_to_image import Scheduler, Optimizations, StepPreviewMode, ImageGenerationResult, _configure_model_padding, model_snapshot_folder, load_pipe
from ..models import Pipeline
from .detect_seamless import SeamlessAxes


def image_to_image(
    self,
    pipeline: Pipeline,
    
    model: str,

    scheduler: Scheduler,

    optimizations: Optimizations,

    image: NDArray,
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

    step_preview_mode: StepPreviewMode,

    # Stability SDK
    key: str | None = None,

    **kwargs
) -> Generator[ImageGenerationResult, None, None]:
    match pipeline:
        case Pipeline.STABLE_DIFFUSION:
            import diffusers
            import torch
            from PIL import Image, ImageOps
            import PIL.Image

            # Mostly copied from `diffusers.StableDiffusionImg2ImgPipeline`, with slight modifications to yield the latents at each step.
            class GeneratorPipeline(diffusers.StableDiffusionImg2ImgPipeline):
                @torch.no_grad()
                def __call__(
                    self,
                    prompt: Union[str, List[str]],
                    image: Union[torch.FloatTensor, PIL.Image.Image],
                    strength: float = 0.8,
                    num_inference_steps: Optional[int] = 50,
                    guidance_scale: Optional[float] = 7.5,
                    negative_prompt: Optional[Union[str, List[str]]] = None,
                    num_images_per_prompt: Optional[int] = 1,
                    eta: Optional[float] = 0.0,
                    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                    output_type: Optional[str] = "pil",
                    return_dict: bool = True,
                    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
                    callback_steps: Optional[int] = 1,
                    **kwargs,
                ):
                    # 1. Check inputs
                    self.check_inputs(prompt, strength, callback_steps)

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

                    # 4. Preprocess image
                    image = diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess(image)

                    # 5. set timesteps
                    self.scheduler.set_timesteps(num_inference_steps, device=device)
                    timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
                    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

                    # 6. Prepare latent variables
                    latents = self.prepare_latents(
                        image, latent_timestep, batch_size, num_images_per_prompt, text_embeddings.dtype, device, generator
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

                    # 9. Post-processing
                    image = self.decode_latents(latents)

                    # TODO: Add UI to enable this
                    # 10. Run safety checker
                    # image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

                    # NOTE: Modified to yield the decoded image as a numpy array.
                    yield ImageGenerationResult(
                        [np.asarray(ImageOps.flip(image).convert('RGBA'), dtype=np.float32) / 255.
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
            pipe = load_pipe(self, "modify", GeneratorPipeline, model, optimizations, scheduler, device)

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
            init_image = Image.fromarray(image).convert('RGB')

            if fit:
                height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
                width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor
                init_image = init_image.resize((width, height))
            else:
                width = init_image.width
                height = init_image.height
            
            # Seamless
            if seamless_axes == SeamlessAxes.AUTO:
                seamless_axes = self.detect_seamless(np.array(init_image) / 255)
            _configure_model_padding(pipe.unet, seamless_axes)
            _configure_model_padding(pipe.vae, seamless_axes)

            # Inference
            with (torch.inference_mode() if device not in ('mps', "privateuseone") else nullcontext()), \
                    (torch.autocast(device) if optimizations.can_use("amp", device) else nullcontext()):
                    yield from pipe(
                        prompt=prompt,
                        image=[init_image] * batch_size,
                        strength=strength,
                        num_inference_steps=steps,
                        guidance_scale=cfg_scale,
                        negative_prompt=negative_prompt if use_negative_prompt else None,
                        num_images_per_prompt=1,
                        eta=0.0,
                        generator=generator,
                        output_type="pil",
                        return_dict=True,
                        callback=None,
                        callback_steps=1,
                        step_preview_mode=step_preview_mode
                    )
        case Pipeline.STABILITY_SDK:
            import stability_sdk.client
            import stability_sdk.interfaces.gooseai.generation.generation_pb2
            from PIL import Image, ImageOps
            import io

            if key is None:
                raise ValueError("DreamStudio key not provided. Enter your key in the add-on preferences.")
            client = stability_sdk.client.StabilityInference(key=key, engine=model)

            if seed is None:
                seed = random.randrange(0, np.iinfo(np.uint32).max)

            answers = client.generate(
                prompt=prompt,
                width=width,
                height=height,
                cfg_scale=cfg_scale,
                sampler=scheduler.stability_sdk(),
                steps=steps,
                seed=seed,
                init_image=(Image.open(image) if isinstance(image, str) else Image.fromarray(image)).convert('RGB'),
                start_schedule=strength,
            )
            for answer in answers:
                for artifact in answer.artifacts:
                    if artifact.finish_reason == stability_sdk.interfaces.gooseai.generation.generation_pb2.FILTER:
                        raise ValueError("Your request activated DreamStudio's safety filter. Please modify your prompt and try again.")
                    if artifact.type == stability_sdk.interfaces.gooseai.generation.generation_pb2.ARTIFACT_IMAGE:
                        image = Image.open(io.BytesIO(artifact.binary))
                        yield ImageGenerationResult(
                            [np.asarray(ImageOps.flip(image).convert('RGBA'), dtype=np.float32) / 255.],
                            [seed],
                            steps,
                            True
                        )
        case _:
            raise Exception(f"Unsupported pipeline {pipeline}.")