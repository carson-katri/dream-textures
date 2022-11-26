from numpy.typing import NDArray
import numpy as np

def upscale(
    self,
    image: str,
    prompt: str,

    half_precision: bool
) -> NDArray:
    import torch
    import diffusers
    from PIL import Image, ImageOps

    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipe = diffusers.StableDiffusionUpscalePipeline.from_pretrained(
        model_id,
        revision="fp16" if half_precision else None,
        torch_dtype=torch.float16 if half_precision else torch.float32
    )
    pipe = pipe.to(self.choose_device())
    
    pipe.enable_attention_slicing()

    result = pipe(prompt=prompt, image=Image.open(image).convert('RGB').resize((128, 128))).images[0]
    return np.asarray(ImageOps.flip(result).convert('RGBA'), dtype=np.float32) / 255.