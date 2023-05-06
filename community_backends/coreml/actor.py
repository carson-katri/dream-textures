import numpy as np
from numpy.typing import NDArray
from dream_textures.generator_process import Actor
from dream_textures.generator_process.future import Future
from dream_textures.generator_process.models import ImageGenerationResult
from dream_textures.api import GenerationResult
import os
import random
import gc

class CoreMLActor(Actor):
    invalidation_args = None
    cached_pipe = None

    def generate(
        self,
        model: str,
        prompt: str,
        negative_prompt: str | None,
        size: tuple[int, int] | None,
        seed: int | None,
        steps: int,
        guidance_scale: float,
        scheduler: str,

        seamless_axes: str,
        step_preview_mode: str,
        iterations: int,

        compute_unit: str,
        controlnet: list[str] | None,
        controlnet_inputs: list[str]
    ):
        future = Future()
        yield future

        import diffusers
        from python_coreml_stable_diffusion import pipeline
        from python_coreml_stable_diffusion import torch2coreml, unet
        import torch
        from PIL import ImageOps
        
        seed = random.randrange(0, np.iinfo(np.uint32).max) if seed is None else seed
        np.random.seed(seed)

        new_invalidation_args = (model, scheduler, controlnet)
        if self.cached_pipe is None or new_invalidation_args != self.invalidation_args:
            self.invalidation_args = new_invalidation_args


            future.add_response(GenerationResult(progress=1, total=1, title="Loading reference pipeline"))

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
                def step_title(i, model_type):
                    future.add_response(GenerationResult(progress=i, total=4, title=f"Converting model to CoreML ({model_type})"))
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

                step_title(1, "VAE decoder")
                torch2coreml.convert_vae_decoder(pytorch_pipe, conversion_args)

                step_title(2, "VAE encoder")
                torch2coreml.convert_vae_encoder(pytorch_pipe, conversion_args)

                step_title(3, "U-Net")
                torch2coreml.convert_unet(pytorch_pipe, conversion_args)

                step_title(4, "text encoder")
                torch2coreml.convert_text_encoder(pytorch_pipe, conversion_args)

            future.add_response(GenerationResult(progress=0, total=1, title=f"Loading converted CoreML pipeline"))

            user_specified_scheduler = None
            if scheduler is not None:
                user_specified_scheduler = pipeline.SCHEDULER_MAP[
                    scheduler.replace(' ', '')].from_config(pytorch_pipe.scheduler.config)

            # NOTE: Modified to have a `callback` parameter.
            def get_coreml_pipe(pytorch_pipe,
                        mlpackages_dir,
                        model_version,
                        compute_unit,
                        delete_original_pipe=True,
                        scheduler_override=None,
                        controlnet_models=None,
                        callback=lambda model_name: None):
                """ Initializes and returns a `CoreMLStableDiffusionPipeline` from an original
                diffusers PyTorch pipeline
                """
                # Ensure `scheduler_override` object is of correct type if specified
                if scheduler_override is not None:
                    assert isinstance(scheduler_override, diffusers.SchedulerMixin)
                    pipeline.logger.warning(
                        "Overriding scheduler in pipeline: "
                        f"Default={pytorch_pipe.scheduler}, Override={scheduler_override}")

                # Gather configured tokenizer and scheduler attributes from the original pipe
                coreml_pipe_kwargs = {
                    "tokenizer": pytorch_pipe.tokenizer,
                    "scheduler": pytorch_pipe.scheduler if scheduler_override is None else scheduler_override,
                    "feature_extractor": pytorch_pipe.feature_extractor,
                }

                model_names_to_load = ["text_encoder", "unet", "vae_decoder"]
                if getattr(pytorch_pipe, "safety_checker", None) is not None:
                    model_names_to_load.append("safety_checker")
                else:
                    pipeline.logger.warning(
                        f"Original diffusers pipeline for {model_version} does not have a safety_checker, "
                        "Core ML pipeline will mirror this behavior.")
                    coreml_pipe_kwargs["safety_checker"] = None

                if delete_original_pipe:
                    del pytorch_pipe
                    gc.collect()
                    pipeline.logger.info("Removed PyTorch pipe to reduce peak memory consumption")

                if controlnet_models:
                    model_names_to_load.remove("unet")
                    callback("control-unet")
                    coreml_pipe_kwargs["unet"] = pipeline._load_mlpackage(
                        "control-unet",
                        mlpackages_dir,
                        model_version,
                        compute_unit,
                    )
                    coreml_pipe_kwargs["controlnet"] = []
                    for i, model_version in enumerate(controlnet_models):
                        callback(f"controlnet-{i}")
                        coreml_pipe_kwargs["controlnet"].append(
                            pipeline._load_mlpackage_controlnet(
                                mlpackages_dir, 
                                model_version, 
                                compute_unit,
                            )
                        )
                else:
                    coreml_pipe_kwargs["controlnet"] = None

                # Load Core ML models
                pipeline.logger.info(f"Loading Core ML models in memory from {mlpackages_dir}")
                def load_package_with_callback(model_name):
                    callback(model_name)
                    return pipeline._load_mlpackage(
                        model_name,
                        mlpackages_dir,
                        model_version,
                        compute_unit,
                    )
                coreml_pipe_kwargs.update({
                    model_name: load_package_with_callback(model_name)
                    for model_name in model_names_to_load
                })
                pipeline.logger.info("Done.")

                pipeline.logger.info("Initializing Core ML pipe for image generation")
                coreml_pipe = pipeline.CoreMLStableDiffusionPipeline(**coreml_pipe_kwargs)
                pipeline.logger.info("Done.")

                return coreml_pipe

            model_i = 1
            def load_callback(model_name):
                nonlocal model_i
                future.add_response(GenerationResult(progress=model_i, total=3 + len(controlnet_inputs), title=f"Loading {model_name} mlpackage (this can take a while)"))
                model_i += 1
            self.cached_pipe = get_coreml_pipe(
                pytorch_pipe=pytorch_pipe,
                mlpackages_dir=mlpackage_dir,
                model_version=model,
                compute_unit=compute_unit,
                scheduler_override=user_specified_scheduler,
                controlnet_models=controlnet,
                callback=load_callback
            )

        height = self.cached_pipe.height if size is None else size[1]
        width = self.cached_pipe.width if size is None else size[0]

        if controlnet:
            controlnet_cond = []
            for i, _ in enumerate(controlnet):
                image_path = controlnet_inputs[i]
                image = pipeline.prepare_controlnet_cond(image_path, height, width)
                controlnet_cond.append(image)
        else:
            controlnet_cond = None

        # Beginning image generation.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        def callback(i, t, latents):
            preview = ImageGenerationResult.step_preview(self, step_preview_mode, width, height, torch.from_numpy(latents), generator, i)
            image = next(iter(preview.images), None)
            future.add_response(GenerationResult(progress=i, total=steps, image=image, seed=seed))
        image = self.cached_pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            controlnet_cond=controlnet_cond,
            negative_prompt=negative_prompt,
            callback=callback
        )

        future.add_response(GenerationResult(progress=steps, total=steps, image=np.asarray(ImageOps.flip(image["images"][0]).convert('RGBA'), dtype=np.float32) / 255., seed=seed))
        future.set_done()