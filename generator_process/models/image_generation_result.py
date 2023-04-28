from typing import List
import math
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
from ...api.models.step_preview_mode import StepPreviewMode

@dataclass
class ImageGenerationResult:
    images: List[NDArray]
    seeds: List[int]
    step: int
    final: bool

    @staticmethod
    def step_preview(pipe, mode, width, height, latents, generator, iteration):
        from PIL import Image, ImageOps
        seeds = [gen.initial_seed() for gen in generator] if isinstance(generator, list) else [generator.initial_seed()]
        match mode:
            case StepPreviewMode.FAST:
                return ImageGenerationResult(
                    [np.asarray(ImageOps.flip(Image.fromarray(approximate_decoded_latents(latents[-1:]))).resize((width, height), Image.Resampling.NEAREST).convert('RGBA'), dtype=np.float32) / 255.],
                    seeds[-1:],
                    iteration,
                    False
                )
            case StepPreviewMode.FAST_BATCH:
                return ImageGenerationResult(
                    [
                        np.asarray(ImageOps.flip(Image.fromarray(approximate_decoded_latents(latents[i:i + 1]))).resize((width, height), Image.Resampling.NEAREST).convert('RGBA'),
                                   dtype=np.float32) / 255.
                        for i in range(latents.size(0))
                    ],
                    seeds,
                    iteration,
                    False
                )
            case StepPreviewMode.ACCURATE:
                return ImageGenerationResult(
                    [np.asarray(ImageOps.flip(pipe.numpy_to_pil(pipe.decode_latents(latents[-1:]))[0]).convert('RGBA'),
                                dtype=np.float32) / 255.],
                    seeds[-1:],
                    iteration,
                    False
                )
            case StepPreviewMode.ACCURATE_BATCH:
                return ImageGenerationResult(
                    [
                        np.asarray(ImageOps.flip(image).convert('RGBA'), dtype=np.float32) / 255.
                        for image in pipe.numpy_to_pil(pipe.decode_latents(latents))
                    ],
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

def approximate_decoded_latents(latents):
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
    latents_ubyte = (((latent_image + 1) / 2)
                    .clamp(0, 1)  # change scale from -1..1 to 0..1
                    .mul(0xFF)  # to 0..255
                    .byte()).cpu()

    return latents_ubyte.numpy()