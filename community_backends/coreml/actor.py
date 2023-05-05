import numpy as np
from numpy.typing import NDArray
from dream_textures.generator_process import Actor
import os

class CoreMLActor(Actor):
    def generate(
        self,
        model: str,
        prompt: str,
        negative_prompt: str | None,
        size: tuple[int, int],
        seed: int,
        steps: int,
        guidance_scale: float,
        scheduler: str,
        seamless_axes: str,
        step_preview_mode: str,
        iterations: int,

        compute_unit: str,
        controlnet: list[str] | None,
        controlnet_inputs: list[str]
    ) -> NDArray:
        from python_coreml_stable_diffusion import pipeline
        from python_coreml_stable_diffusion import torch2coreml, unet

        np.random.seed(seed)

        # Initializing PyTorch pipe for reference configuration
        from diffusers import StableDiffusionPipeline
        pytorch_pipe = StableDiffusionPipeline.from_pretrained(model,
                                                            use_auth_token=True)
        # There is currently no UI for this, so remove it.
        # This avoids wasting time converting and loading it.
        pytorch_pipe.safety_checker = None
        
        mlpackage_cache = os.path.expanduser(
            os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "dream_textures_coreml"))
        )
        mlpackage_dir = os.path.join(mlpackage_cache, model.replace('/', '_'))

        if not os.path.exists(mlpackage_dir):
            os.makedirs(mlpackage_dir, exist_ok=True)
            class ConversionArgs:
                model_version = model
                compute_unit = 'ALL'
                latent_h = None
                latent_w = None
                attention_implementation = unet.ATTENTION_IMPLEMENTATION_IN_EFFECT.name
                o = mlpackage_dir
                check_output_correctness = False
                chunk_unet = False
                quantize_weights_to_8bits = False
                unet_support_controlnet = False
                text_encoder_vocabulary_url = "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json"
                text_encoder_merges_url = "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt"
            conversion_args = ConversionArgs()
            torch2coreml.convert_vae_decoder(pytorch_pipe, conversion_args)
            print("VAE decoder converted")
            torch2coreml.convert_vae_encoder(pytorch_pipe, conversion_args)
            print("VAE encoder converted")
            torch2coreml.convert_unet(pytorch_pipe, conversion_args)
            print("U-Net converted")
            torch2coreml.convert_text_encoder(pytorch_pipe, conversion_args)
            print("Text encoder converted")

        user_specified_scheduler = None
        if scheduler is not None:
            user_specified_scheduler = pipeline.SCHEDULER_MAP[
                scheduler.replace(' ', '')].from_config(pytorch_pipe.scheduler.config)

        coreml_pipe = pipeline.get_coreml_pipe(
            pytorch_pipe=pytorch_pipe,
            mlpackages_dir=mlpackage_dir,
            model_version=model,
            compute_unit=compute_unit,
            scheduler_override=user_specified_scheduler,
            controlnet_models=controlnet
        )

        if controlnet:
            controlnet_cond = []
            for i, _ in enumerate(controlnet):
                image_path = controlnet_inputs[i]
                image = pipeline.prepare_controlnet_cond(image_path, coreml_pipe.height, coreml_pipe.width)
                controlnet_cond.append(image)
        else:
            controlnet_cond = None

        # Beginning image generation.
        image = coreml_pipe(
            prompt=prompt,
            height=coreml_pipe.height,
            width=coreml_pipe.width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            controlnet_cond=controlnet_cond,
            negative_prompt=negative_prompt,
        )

        image["images"][0].save('test.png')
        return image["images"][0]