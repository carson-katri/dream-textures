from typing import Union, Generator, Callable, List, Optional
import os
from contextlib import nullcontext
from numpy.typing import NDArray
import numpy as np
import random
from .prompt_to_image import Pipeline, Scheduler, Optimizations, StepPreviewMode, ImageGenerationResult, _configure_model_padding, model_snapshot_folder, load_pipe
from .detect_seamless import SeamlessAxes

def inpaint(
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

    iterations: int,

    step_preview_mode: StepPreviewMode,

    inpaint_mask_src: str,
    text_mask: str,
    text_mask_confidence: float,

    # Stability SDK
    key: str | None = None,

    **kwargs
) -> Generator[NDArray, None, None]:
    match pipeline:
        case Pipeline.STABLE_DIFFUSION:
            import diffusers
            import torch
            from PIL import Image, ImageOps
            import PIL.Image

            # Mostly copied from `diffusers.StableDiffusionInpaintPipeline`, with slight modifications to yield the latents at each step.
            class GeneratorPipeline(diffusers.StableDiffusionInpaintPipeline):
                @torch.no_grad()
                def __call__(
                    self,
                    prompt: Union[str, List[str]],
                    image: Union[torch.FloatTensor, PIL.Image.Image],
                    mask_image: Union[torch.FloatTensor, PIL.Image.Image],
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

                    # 4. Preprocess mask and image
                    mask, masked_image = diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint.prepare_mask_and_masked_image(image, mask_image)

                    # 5. set timesteps
                    self.scheduler.set_timesteps(num_inference_steps, device=device)
                    timesteps = self.scheduler.timesteps

                    # 6. Prepare latent variables
                    num_channels_latents = self.vae.config.latent_channels
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
                    mask, masked_image_latents = self.prepare_mask_latents(
                        mask,
                        masked_image,
                        batch_size * num_images_per_prompt,
                        height,
                        width,
                        text_embeddings.dtype,
                        device,
                        generator,
                        do_classifier_free_guidance,
                    )

                    # 8. Check that sizes of mask, masked image and latents match
                    num_channels_mask = mask.shape[1]
                    num_channels_masked_image = masked_image_latents.shape[1]
                    if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
                        raise ValueError(
                            f"Select an inpainting model, such as 'stabilityai/stable-diffusion-2-inpainting'"
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
                            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

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
            pipe = load_pipe(self, "inpaint", GeneratorPipeline, model, optimizations, scheduler, device)

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
            init_image = Image.fromarray(image)
            
            # Seamless
            if seamless_axes == SeamlessAxes.AUTO:
                seamless_axes = self.detect_seamless(np.array(init_image) / 255)
            _configure_model_padding(pipe.unet, seamless_axes)
            _configure_model_padding(pipe.vae, seamless_axes)

            # Inference
            with (torch.inference_mode() if device not in ('mps', "privateuseone") else nullcontext()), \
                    (torch.autocast(device) if optimizations.can_use("amp", device) else nullcontext()):
                    match inpaint_mask_src:
                        case 'alpha':
                            mask_image = ImageOps.invert(init_image.getchannel('A'))
                        case 'prompt':
                            from transformers import AutoProcessor, CLIPSegForImageSegmentation

                            processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
                            clipseg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
                            inputs = processor(text=[text_mask], images=[init_image.convert('RGB')], return_tensors="pt", padding=True)
                            outputs = clipseg(**inputs)
                            mask_image = Image.fromarray(np.uint8((1 - torch.sigmoid(outputs.logits).lt(text_mask_confidence).int().detach().numpy()) * 255), 'L').resize(init_image.size)

                    yield from pipe(
                        prompt=prompt,
                        image=[init_image.convert('RGB')] * batch_size,
                        mask_image=[mask_image] * batch_size,
                        strength=strength,
                        height=init_image.size[1] if fit else height,
                        width=init_image.size[0] if fit else width,
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
            import stability_sdk.client
            import stability_sdk.interfaces.gooseai.generation.generation_pb2
            from PIL import Image, ImageOps
            import io

            if key is None:
                raise ValueError("DreamStudio key not provided. Enter your key in the add-on preferences.")
            client = stability_sdk.client.StabilityInference(key=key, engine=model)

            if seed is None:
                seed = random.randrange(0, np.iinfo(np.uint32).max)

            init_image = Image.open(image) if isinstance(image, str) else Image.fromarray(image)

            answers = client.generate(
                prompt=prompt,
                width=width or 512,
                height=height or 512,
                cfg_scale=cfg_scale,
                sampler=scheduler.stability_sdk(),
                steps=steps,
                seed=seed,
                init_image=init_image.convert('RGB'),
                mask_image=init_image.getchannel('A'),
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