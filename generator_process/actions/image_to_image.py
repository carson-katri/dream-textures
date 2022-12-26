from typing import Union, Generator, Callable, List, Optional
import os
from contextlib import nullcontext

from numpy.typing import NDArray
import numpy as np
import random
from .prompt_to_image import Pipeline, Scheduler, Optimizations, StepPreviewMode, ImageGenerationResult, approximate_decoded_latents, _configure_model_padding
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
    prompt: str,
    steps: int,
    width: int,
    height: int,
    seed: int,

    cfg_scale: float,
    use_negative_prompt: bool,
    negative_prompt: str,
    
    seamless_axes: SeamlessAxes | str | bool | tuple[bool, bool] | None,

    iterations: int,

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
                    generator: Optional[torch.Generator] = None,
                    output_type: Optional[str] = "pil",
                    return_dict: bool = True,
                    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
                    callback_steps: Optional[int] = 1,
                    **kwargs,
                ):
                    image = init_image or image

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
                    if isinstance(image, PIL.Image.Image):
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

                            # NOTE: Modified to yield the latents instead of calling a callback.
                            match kwargs['step_preview_mode']:
                                case StepPreviewMode.NONE:
                                    yield ImageGenerationResult(
                                        None,
                                        generator.initial_seed(),
                                        i,
                                        False
                                    )
                                case StepPreviewMode.FAST:
                                    yield ImageGenerationResult(
                                        np.asarray(ImageOps.flip(Image.fromarray(approximate_decoded_latents(latents))).resize((width, height), Image.Resampling.NEAREST).convert('RGBA'), dtype=np.float32) / 255.,
                                        generator.initial_seed(),
                                        i,
                                        False
                                    )
                                case StepPreviewMode.ACCURATE:
                                    yield from [
                                        ImageGenerationResult(
                                            np.asarray(ImageOps.flip(image).convert('RGBA'), dtype=np.float32) / 255.,
                                            generator.initial_seed(),
                                            i,
                                            False
                                        )
                                        for image in self.numpy_to_pil(self.decode_latents(latents))
                                    ]

                    # 9. Post-processing
                    image = self.decode_latents(latents)

                    # TODO: Add UI to enable this
                    # 10. Run safety checker
                    # image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

                    # NOTE: Modified to yield the decoded image as a numpy array.
                    yield from [
                        ImageGenerationResult(
                            np.asarray(ImageOps.flip(image).convert('RGBA'), dtype=np.float32) / 255.,
                            generator.initial_seed(),
                            num_inference_steps,
                            True
                        )
                        for image in self.numpy_to_pil(image)
                    ]
            
            if optimizations.cpu_only:
                device = "cpu"
            else:
                device = self.choose_device()

            use_cpu_offload = optimizations.can_use("sequential_cpu_offload", device)

            # StableDiffusionPipeline w/ caching
            if hasattr(self, "_cached_img2img_pipe") and self._cached_img2img_pipe[1] == model and use_cpu_offload == self._cached_img2img_pipe[2]:
                pipe = self._cached_img2img_pipe[0]
            else:
                storage_folder = model
                if os.path.exists(os.path.join(storage_folder, 'model_index.json')):
                    snapshot_folder = storage_folder
                else:
                    revision = "main"
                    ref_path = os.path.join(storage_folder, "refs", revision)
                    with open(ref_path) as f:
                        commit_hash = f.read()

                    snapshot_folder = os.path.join(storage_folder, "snapshots", commit_hash)
                pipe = GeneratorPipeline.from_pretrained(
                    snapshot_folder,
                    revision="fp16" if optimizations.can_use("half_precision", device) else None,
                    torch_dtype=torch.float16 if optimizations.can_use("half_precision", device) else torch.float32,
                )
                pipe = pipe.to(device)
                setattr(self, "_cached_img2img_pipe", (pipe, model, use_cpu_offload, snapshot_folder))

            # Scheduler
            is_stable_diffusion_2 = 'stabilityai--stable-diffusion-2' in model
            pipe.scheduler = scheduler.create(pipe, {
                'model_path': self._cached_img2img_pipe[3],
                'subfolder': 'scheduler',
            } if is_stable_diffusion_2 else None)

            # Optimizations
            pipe = optimizations.apply(pipe, device)

            # RNG
            generator = torch.Generator(device="cpu" if device == "mps" else device) # MPS does not support the `Generator` API
            if seed is None:
                seed = random.randrange(0, np.iinfo(np.uint32).max)
            generator = generator.manual_seed(seed)

            # Init Image
            init_image = Image.fromarray(image).convert('RGB')
            
            # Seamless
            if seamless_axes == SeamlessAxes.AUTO:
                seamless_axes = self.detect_seamless(np.array(init_image) / 255)
            _configure_model_padding(pipe.unet, seamless_axes)
            _configure_model_padding(pipe.vae, seamless_axes)

            # Inference
            with (torch.inference_mode() if device != 'mps' else nullcontext()), \
                    (torch.autocast(device) if optimizations.can_use("amp", device) else nullcontext()):
                    yield from pipe(
                        prompt=prompt,
                        image=init_image if fit else init_image.resize((width, height)),
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
                            np.asarray(ImageOps.flip(image).convert('RGBA'), dtype=np.float32) / 255.,
                            seed,
                            steps,
                            True
                        )
        case _:
            raise Exception(f"Unsupported pipeline {pipeline}.")