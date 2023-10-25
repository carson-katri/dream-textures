from typing import List
from dataclasses import dataclass
from numpy.typing import NDArray
from ...api.models.step_preview_mode import StepPreviewMode

# TODO: replace class use with lists of api.models.generation_result.GenerationResult, keep step_preview() and decoding here
@dataclass
class ImageGenerationResult:
    images: List[NDArray]
    seeds: List[int]
    step: int
    final: bool
    total: int | None = None

    @staticmethod
    def step_preview(pipe, mode, width, height, latents, generator, iteration):
        seeds = [gen.initial_seed() for gen in generator] if isinstance(generator, list) else [generator.initial_seed()]
        match mode:
            case StepPreviewMode.FAST:
                return ImageGenerationResult(
                    [approximate_decoded_latents(latents[-1:], (height, width))],
                    seeds[-1:],
                    iteration,
                    False
                )
            case StepPreviewMode.FAST_BATCH:
                return ImageGenerationResult(
                    [
                        approximate_decoded_latents(latents[i:i + 1], (height, width))
                        for i in range(latents.size(0))
                    ],
                    seeds,
                    iteration,
                    False
                )
            case StepPreviewMode.ACCURATE:
                return ImageGenerationResult(
                    list(decode_latents(pipe, latents[-1:])),
                    seeds[-1:],
                    iteration,
                    False
                )
            case StepPreviewMode.ACCURATE_BATCH:
                return ImageGenerationResult(
                    list(decode_latents(pipe, latents)),
                    seeds,
                    iteration,
                    False
                )
        return ImageGenerationResult(
            [],
            seeds,
            iteration,
            False
        )

def decode_latents(pipe, latents):
    return pipe.image_processor.postprocess(pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample, output_type="np")

def approximate_decoded_latents(latents, size=None):
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
    if size is not None:
        latent_image = torch.nn.functional.interpolate(
            latent_image.permute(2, 0, 1).unsqueeze(0), size=size, mode="nearest"
        ).squeeze(0).permute(1, 2, 0)
    latent_image = ((latent_image + 1) / 2).clamp(0, 1).cpu()
    return latent_image.numpy()
