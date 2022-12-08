from typing import Union, Generator, Callable, List, Optional, Dict, Tuple
import os
from contextlib import nullcontext
from numpy.typing import NDArray
import numpy as np
import random
import math
from .prompt_to_image import Pipeline, Scheduler, Optimizations, StepPreviewMode, approximate_decoded_latents, ImageGenerationResult

def depth_to_image(
    self,
    pipeline: Pipeline,
    
    model: str,

    scheduler: Scheduler,

    optimizations: Optimizations,

    depth: NDArray | str,
    image: NDArray | str | None,
    segmentation_map: NDArray | str | None,
    segmentation_prompts: Dict[int, str],
    strength: float,
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
                depth = torch.from_numpy(depth)
                return depth
            
            def _img_importance_flatten(img: torch.Tensor, ratio: int) -> torch.Tensor:
                return torch.nn.functional.interpolate(
                    img.unsqueeze(0).unsqueeze(1),
                    scale_factor=1 / ratio,
                    mode="bilinear",
                    align_corners=True,
                ).squeeze()

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

                def prepare_img2img_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None, image=None, timestep=None):
                    shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
                    if latents is None:
                        if device.type == "mps":
                            # randn does not work reproducibly on mps
                            latents = torch.randn(shape, generator=generator, device="cpu", dtype=dtype).to(device)
                        else:
                            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
                    else:
                        if latents.shape != shape:
                            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
                        latents = latents.to(device)

                    # scale the initial noise by the standard deviation required by the scheduler
                    latents = latents * self.scheduler.init_noise_sigma

                    if image is not None:
                        image = image.to(device=device, dtype=dtype)
                        image_latents = self.vae.encode(image).latent_dist.sample(generator=generator)
                        image_latents = torch.nn.functional.interpolate(
                            image_latents, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
                        )
                        image_latents = 0.18215 * image_latents
                        noise = torch.randn(image_latents.shape, generator=generator, device=device, dtype=dtype)
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

                #region Segmentation
                def _segmentation_encode_prompts(
                    self,
                    segmentation_map: PIL.Image.Image,
                    segmentation_prompts: Dict[int, str]
                ):
                    tokenized_segments = []

                    for segment_color, segment_prompt in segmentation_prompts.items():
                        f = 1.5
                        v_input = self.tokenizer(
                            segment_prompt,
                            max_length=self.tokenizer.model_max_length,
                            truncation=True,
                        )
                        v_as_tokens = v_input["input_ids"][1:-1]

                        img_where_color = (np.array(segmentation_map) == segment_color)

                        img_where_color = torch.tensor(img_where_color, dtype=torch.float32) * f

                        tokenized_segments.append((v_as_tokens, img_where_color))

                    if len(tokenized_segments) == 0:
                        tokenized_segments.append(([-1], torch.zeros(segmentation_map.size[:2], dtype=torch.float32)))
                    return tokenized_segments
                
                def _segmentation_tokens_weight(
                    self,
                    encoded_prompts,
                    tokenized,
                    ratio
                ):
                    token_lis = tokenized["input_ids"][0].tolist()
                    w, h = encoded_prompts[0][1].shape

                    w_r, h_r = w // ratio, h // ratio

                    weights = torch.zeros((w_r * h_r, len(token_lis)), dtype=torch.float32)

                    for v_as_tokens, img_where_color in encoded_prompts:
                        is_in = 0

                        for i in range(len(token_lis)):
                            if token_lis[i : i + len(v_as_tokens)] == v_as_tokens:
                                is_in = 1
                                weights[:, i : i + len(v_as_tokens)] += _img_importance_flatten(img_where_color, ratio).reshape(-1, 1).repeat(1, len(v_as_tokens))

                        if not is_in == 1:
                            print(f"Warning ratio {ratio} : tokens {v_as_tokens} not found in text")

                    return weights
                #endregion

                @torch.no_grad()
                def __call__(
                    self,
                    prompt: Union[str, List[str]],
                    depth_image: Union[torch.FloatTensor, PIL.Image.Image],
                    image: Optional[Union[torch.FloatTensor, PIL.Image.Image]] = None,
                    segmentation_map: Optional[Union[torch.FloatTensor, PIL.Image.Image]] = None,
                    segmentation_prompts: Optional[Dict[int, str]] = None,
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
                    _configure_paint_with_words_attention(self.unet)

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

                    # 3b. Segmentation
                    text_inputs = self.tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    segmented_embeddings = self._segmentation_encode_prompts(segmentation_map, segmentation_prompts)
                    cross_attention_weights = torch.cat([self._segmentation_tokens_weight(segmented_embeddings, text_inputs, 8 * (2 ** ratio)) for ratio in range(4)]).to(device)

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

                            # NOTE: Segmentation passed through encoder_hidden_states
                            if segmentation_map is not None:
                                encoder_hidden_states = [text_embeddings, cross_attention_weights, self.scheduler.sigmas[i]]
                            else:
                                encoder_hidden_states = text_embeddings

                            # predict the noise residual
                            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states).sample

                            # perform guidance
                            if do_classifier_free_guidance:
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                            # compute the previous noisy sample x_t -> x_t-1
                            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                            # call the callback, if provided
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
                                        np.asarray(PIL.ImageOps.flip(PIL.Image.fromarray(approximate_decoded_latents(latents))).resize(final_size, PIL.Image.Resampling.NEAREST).convert('RGBA'), dtype=np.float32) / 255.,
                                        generator.initial_seed(),
                                        i,
                                        False
                                    )
                                case StepPreviewMode.ACCURATE:
                                    yield from [
                                        ImageGenerationResult(
                                            np.asarray(PIL.ImageOps.flip(image).resize(final_size).convert('RGBA'), dtype=np.float32) / 255.,
                                            generator.initial_seed(),
                                            i,
                                            False
                                        )
                                        for image in self.numpy_to_pil(self.decode_latents(latents))
                                    ]

                    # 11. Post-processing
                    image = self.decode_latents(latents)

                    # TODO: Add UI to enable this.
                    # 12. Run safety checker
                    # image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)

                    # NOTE: Modified to yield the decoded image as a numpy array.
                    yield from [
                        ImageGenerationResult(
                            np.asarray(PIL.ImageOps.flip(image).resize(final_size).convert('RGBA'), dtype=np.float32) / 255.,
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
            generator = torch.Generator(device="cpu" if device == "mps" else device) # MPS does not support the `Generator` API
            if seed is None:
                seed = random.randrange(0, np.iinfo(np.uint32).max)
            generator = generator.manual_seed(seed)

            # Inference
            rounded_size = (
                int(8 * (depth.shape[1] // 8)),
                int(8 * (depth.shape[0] // 8)),
            )
            with (torch.inference_mode() if device != 'mps' else nullcontext()), \
                (torch.autocast(device) if optimizations.can_use("amp", device) else nullcontext()):
                depth_image = PIL.ImageOps.flip(PIL.Image.fromarray(np.uint8(depth * 255), 'L')).resize(rounded_size)
                init_image = None if image is None else (PIL.Image.open(image) if isinstance(image, str) else PIL.Image.fromarray(image.astype(np.uint8))).convert('RGB').resize(rounded_size)
                if segmentation_map is not None:
                    segmentation_map = PIL.ImageOps.flip(PIL.Image.fromarray(np.uint8(segmentation_map * 255), 'L')).resize(rounded_size)
                yield from pipe(
                    prompt=prompt,
                    depth_image=depth_image,
                    image=init_image,
                    segmentation_map=segmentation_map,
                    segmentation_prompts=segmentation_prompts,
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

def _configure_paint_with_words_attention(unet):
    import torch

    def forward(
        self,
        hidden_states,
        context=None,
        mask=None
    ):
        """Paint with words attention based on https://github.com/cloneofsimo/paint-with-words-sd

        `context` is expected to be in the following format:
        ```python
        [text_embeddings, [weight_64, weight_256, weight_1024, weight_4096], sigma]
        ```
        """
        
        if context is not None:
            context_tensor = context[0]
        else:
            context_tensor = hidden_states

        query = self.to_q(hidden_states)

        key = self.to_k(context_tensor)
        value = self.to_v(context_tensor)

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))

        attention_size_of_img = attention_scores.shape[-2]
        attention_weight_index = int(math.log2(attention_size_of_img // 64) // 2)
        if context is not None:
            w = context[1][attention_weight_index]
            sigma = context[2]

            cross_attention_weight = 0.1 * w * math.log(sigma + 1) * attention_scores.max()
        else:
            cross_attention_weight = 0.0

        attention_scores = (attention_scores + cross_attention_weight) * self.scale

        attention_probs = attention_scores.softmax(dim=-1)

        hidden_states = torch.matmul(attention_probs, value)

        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states
    
    for _module in unet.modules():
        if _module.__class__.__name__ == "CrossAttention":
            _module.__class__.__call__ = forward