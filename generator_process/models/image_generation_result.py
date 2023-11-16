from ...api.models.step_preview_mode import StepPreviewMode
from ...api.models.generation_result import GenerationResult

def step_latents(pipe, mode, latents, generator, iteration, steps):
    seeds = [gen.initial_seed() for gen in generator] if isinstance(generator, list) else [generator.initial_seed()]
    scale = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    match mode:
        case StepPreviewMode.FAST:
            return [
                GenerationResult(
                    progress=iteration,
                    total=steps,
                    seed=seeds[-1],
                    image=approximate_decoded_latents(latents[-1:], scale)
                )
            ]
        case StepPreviewMode.FAST_BATCH:
            return [
                GenerationResult(
                    progress=iteration,
                    total=steps,
                    seed=seed,
                    image=approximate_decoded_latents(latent, scale)
                )
                for latent, seed in zip(latents[:, None], seeds)
            ]
        case StepPreviewMode.ACCURATE:
            return [
                GenerationResult(
                    progress=iteration,
                    total=steps,
                    seed=seeds[-1],
                    image=decode_latents(pipe, latents[-1:])
                )
            ]
        case StepPreviewMode.ACCURATE_BATCH:
            return [
                GenerationResult(
                    progress=iteration,
                    total=steps,
                    seed=seed,
                    image=decode_latents(pipe, latent)
                )
                for latent, seed in zip(latents[:, None], seeds)
            ]
    return [
        GenerationResult(
            progress=iteration,
            total=steps,
            seed=seeds[-1]
        )
    ]

def step_images(images, generator, iteration, steps):
    if not isinstance(images, list) and images.ndim == 3:
        images = images[None]
    seeds = [gen.initial_seed() for gen in generator] if isinstance(generator, list) else [generator.initial_seed()]
    return [
        GenerationResult(
            progress=iteration,
            total=steps,
            seed=seed,
            image=image
        )
        for image, seed in zip(images, seeds)
    ]

def decode_latents(pipe, latents):
    return pipe.image_processor.postprocess(pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample, output_type="np")

def approximate_decoded_latents(latents, scale=None):
    """
    Approximate the decoded latents without using the VAE.
    """
    import torch
    # origingally adapted from code by @erucipe and @keturn here:
    # https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/7

    # these updated numbers for v1.5 are from @torridgristle
    v1_5_latent_rgb_factors = torch.tensor([
        #    R        G        B
        [ 0.3444,  0.1385,  0.0670], # L1
        [ 0.1247,  0.4027,  0.1494], # L2
        [-0.3192,  0.2513,  0.2103], # L3
        [-0.1307, -0.1874, -0.7445]  # L4
    ], dtype=latents.dtype, device=latents.device)

    latent_image = latents[0].permute(1, 2, 0) @ v1_5_latent_rgb_factors
    if scale is not None:
        latent_image = torch.nn.functional.interpolate(
            latent_image.permute(2, 0, 1).unsqueeze(0), scale_factor=scale, mode="nearest"
        ).squeeze(0).permute(1, 2, 0)
    latent_image = ((latent_image + 1) / 2).clamp(0, 1).cpu()
    return latent_image.numpy()
