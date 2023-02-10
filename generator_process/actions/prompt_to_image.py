from typing import Annotated, Union, _AnnotatedAlias, Generator, Callable, List, Optional, Any
import enum
import math
import os
import sys
from dataclasses import dataclass
from contextlib import nullcontext

from numpy.typing import NDArray
import numpy as np
import random
from .detect_seamless import SeamlessAxes
from ...absolute_path import absolute_path

from ..models import Pipeline

class CachedPipeline:
    """A pipeline that has been cached for subsequent runs."""

    pipeline: Any
    """The diffusers pipeline to re-use"""

    invalidation_properties: tuple
    """Values that, when changed, will invalid this cached pipeline"""

    snapshot_folder: str
    """The snapshot folder containing the model"""

    def __init__(self, pipeline: Any, invalidation_properties: tuple, snapshot_folder: str):
        self.pipeline = pipeline
        self.invalidation_properties = invalidation_properties
        self.snapshot_folder = snapshot_folder

    def is_valid(self, properties: tuple):
        return properties == self.invalidation_properties

def load_pipe(self, action, generator_pipeline, model, optimizations, scheduler, device):
    """
    Use a cached pipeline, or create the pipeline class and cache it.
    
    The cached pipeline will be invalidated if the model or use_cpu_offload options change.
    """
    import torch
    import gc

    invalidation_properties = (
        action, model, device,
        optimizations.can_use("sequential_cpu_offload", device),
        optimizations.can_use("half_precision", device),
    )
    cached_pipe: CachedPipeline = self._cached_pipe if hasattr(self, "_cached_pipe") else None
    if cached_pipe is not None and cached_pipe.is_valid(invalidation_properties):
        pipe = cached_pipe.pipeline
    else:
        # Release the cached pipe before loading the new one.
        if cached_pipe is not None:
            del self._cached_pipe
            del cached_pipe
            gc.collect()

        revision = "fp16" if optimizations.can_use_half(device) else None
        snapshot_folder = model_snapshot_folder(model, revision)
        pipe = generator_pipeline.from_pretrained(
            snapshot_folder,
            revision=revision,
            torch_dtype=torch.float16 if optimizations.can_use_half(device) else torch.float32,
        )
        pipe = pipe.to(device)
        setattr(self, "_cached_pipe", CachedPipeline(pipe, invalidation_properties, snapshot_folder))
        cached_pipe = self._cached_pipe
    if 'scheduler' in os.listdir(cached_pipe.snapshot_folder):
        pipe.scheduler = scheduler.create(pipe, {
            'model_path': cached_pipe.snapshot_folder,
            'subfolder': 'scheduler',
        })
    else:
        pipe.scheduler = scheduler.create(pipe, None)
    return pipe

class Scheduler(enum.Enum):
    DDIM = "DDIM"
    DDPM = "DDPM"
    DEIS_MULTISTEP = "DEIS Multistep"
    DPM_SOLVER_MULTISTEP = "DPM Solver Multistep"
    DPM_SOLVER_SINGLESTEP = "DPM Solver Singlestep"
    EULER_DISCRETE = "Euler Discrete"
    EULER_ANCESTRAL_DISCRETE = "Euler Ancestral Discrete"
    HEUN_DISCRETE = "Heun Discrete"
    KDPM2_DISCRETE = "KDPM2 Discrete" # Non-functional on mps
    KDPM2_ANCESTRAL_DISCRETE = "KDPM2 Ancestral Discrete"
    LMS_DISCRETE = "LMS Discrete"
    PNDM = "PNDM"

    def create(self, pipeline, pretrained):
        import diffusers
        def scheduler_class():
            match self:
                case Scheduler.DDIM:
                    return diffusers.schedulers.DDIMScheduler
                case Scheduler.DDPM:
                    return diffusers.schedulers.DDPMScheduler
                case Scheduler.DEIS_MULTISTEP:
                    return diffusers.schedulers.DEISMultistepScheduler
                case Scheduler.DPM_SOLVER_MULTISTEP:
                    return diffusers.schedulers.DPMSolverMultistepScheduler
                case Scheduler.DPM_SOLVER_SINGLESTEP:
                    return diffusers.schedulers.DPMSolverSinglestepScheduler
                case Scheduler.EULER_DISCRETE:
                    return diffusers.schedulers.EulerDiscreteScheduler
                case Scheduler.EULER_ANCESTRAL_DISCRETE:
                    return diffusers.schedulers.EulerAncestralDiscreteScheduler
                case Scheduler.HEUN_DISCRETE:
                    return diffusers.schedulers.HeunDiscreteScheduler
                case Scheduler.KDPM2_DISCRETE:
                    return diffusers.schedulers.KDPM2DiscreteScheduler
                case Scheduler.KDPM2_ANCESTRAL_DISCRETE:
                    return diffusers.schedulers.KDPM2AncestralDiscreteScheduler
                case Scheduler.LMS_DISCRETE:
                    return diffusers.schedulers.LMSDiscreteScheduler
                case Scheduler.PNDM:
                    return diffusers.schedulers.PNDMScheduler
        if pretrained is not None:
            return scheduler_class().from_pretrained(pretrained['model_path'], subfolder=pretrained['subfolder'])
        else:
            return scheduler_class().from_config(pipeline.scheduler.config)
    
    def stability_sdk(self):
        import stability_sdk.interfaces.gooseai.generation.generation_pb2
        match self:
            case Scheduler.LMS_DISCRETE:
                return stability_sdk.interfaces.gooseai.generation.generation_pb2.SAMPLER_K_LMS
            case Scheduler.DDIM:
                return stability_sdk.interfaces.gooseai.generation.generation_pb2.SAMPLER_DDIM
            case Scheduler.DDPM:
                return stability_sdk.interfaces.gooseai.generation.generation_pb2.SAMPLER_DDPM
            case Scheduler.EULER_DISCRETE:
                return stability_sdk.interfaces.gooseai.generation.generation_pb2.SAMPLER_K_EULER
            case Scheduler.EULER_ANCESTRAL_DISCRETE:
                return stability_sdk.interfaces.gooseai.generation.generation_pb2.SAMPLER_K_EULER_ANCESTRAL
            case _:
                raise ValueError(f"{self} cannot be used with DreamStudio.")

@dataclass(eq=True)
class Optimizations:
    attention_slicing: bool = True
    attention_slice_size: Union[str, int] = "auto"
    cudnn_benchmark: Annotated[bool, "cuda"] = False
    tf32: Annotated[bool, "cuda"] = False
    amp: Annotated[bool, "cuda"] = False
    half_precision: Annotated[bool, {"cuda", "privateuseone"}] = True
    sequential_cpu_offload: Annotated[bool, {"cuda", "privateuseone"}] = False
    channels_last_memory_format: bool = False
    # xformers_attention: bool = False # FIXME: xFormers is not yet available.
    batch_size: int = 1
    vae_slicing: bool = True

    cpu_only: bool = False

    @staticmethod
    def infer_device() -> str:
        if sys.platform == "darwin":
            return "mps"
        elif Pipeline.directml_available():
            return "privateuseone"
        else:
            return "cuda"

    def can_use(self, property, device) -> bool:
        if not getattr(self, property):
            return False
        if isinstance(self.__annotations__.get(property, None), _AnnotatedAlias):
            annotation: _AnnotatedAlias = self.__annotations__[property]
            opt_dev = annotation.__metadata__[0]
            if isinstance(opt_dev, str):
                return opt_dev == device
            return device in opt_dev
        return True

    def can_use_half(self, device):
        if self.half_precision and device == "cuda":
            import torch
            name = torch.cuda.get_device_name()
            return not ("GTX 1650" in name or "GTX 1660" in name)
        return self.can_use("half_precision", device)
    
    def apply(self, pipeline, device):
        """
        Apply the optimizations to a diffusers pipeline.

        All exceptions are ignored to make this more general purpose across different pipelines.
        """
        import torch

        torch.backends.cudnn.benchmark = self.can_use("cudnn_benchmark", device)
        torch.backends.cuda.matmul.allow_tf32 = self.can_use("tf32", device)

        try:
            if self.can_use("attention_slicing", device):
                pipeline.enable_attention_slicing(self.attention_slice_size)
            else:
                pipeline.disable_attention_slicing()
        except: pass
        
        try:
            if self.can_use("sequential_cpu_offload", device) and device in ["cuda", "privateuseone"]:
                # Doesn't allow for selecting execution device
                # pipeline.enable_sequential_cpu_offload()

                from accelerate import cpu_offload

                for cpu_offloaded_model in [pipeline.unet, pipeline.text_encoder, pipeline.vae]:
                    if cpu_offloaded_model is not None:
                        cpu_offload(cpu_offloaded_model, device)

                if pipeline.safety_checker is not None:
                    cpu_offload(pipeline.safety_checker.vision_model, device)
        except: pass
        
        try:
            if self.can_use("channels_last_memory_format", device):
                pipeline.unet.to(memory_format=torch.channels_last)
            else:
                pipeline.unet.to(memory_format=torch.contiguous_format)
        except: pass

        # FIXME: xFormers wheels are not yet available (https://github.com/facebookresearch/xformers/issues/533)
        # if self.can_use("xformers_attention", device):
        #     pipeline.enable_xformers_memory_efficient_attention()

        try:
            if self.can_use("vae_slicing", device):
                # Not many pipelines implement the enable_vae_slicing()/disable_vae_slicing()
                # methods but all they do is forward their call to the vae anyway.
                pipeline.vae.enable_slicing()
            else:
                pipeline.vae.disable_slicing()
        except: pass
        
        from .. import directml_patches
        if device == "privateuseone":
            directml_patches.enable(pipeline)
        else:
            directml_patches.disable(pipeline)

        return pipeline

class StepPreviewMode(enum.Enum):
    NONE = "None"
    FAST = "Fast"
    FAST_BATCH = "Fast (Batch Tiled)"
    ACCURATE = "Accurate"
    ACCURATE_BATCH = "Accurate (Batch Tiled)"

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
            [seeds],
            iteration,
            False
        )

    def tile_images(self):
        images = self.images
        if len(images) == 0:
            return None
        elif len(images) == 1:
            return images[0]
        width = images[0].shape[1]
        height = images[0].shape[0]
        tiles_x = math.ceil(math.sqrt(len(images)))
        tiles_y = math.ceil(len(images) / tiles_x)
        tiles = np.zeros((height * tiles_y, width * tiles_x, 4), dtype=np.float32)
        bottom_offset = (tiles_x*tiles_y-len(images)) * width // 2
        for i, image in enumerate(images):
            x = i % tiles_x
            y = tiles_y - 1 - int((i - x) / tiles_x)
            x *= width
            y *= height
            if y == 0:
                x += bottom_offset
            tiles[y: y + height, x: x + width] = image
        return tiles

def choose_device(self) -> str:
    """
    Automatically select which PyTorch device to use.
    """
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    if Pipeline.directml_available():
        import torch_directml
        if torch_directml.is_available():
            # can be named better when torch.utils.rename_privateuse1_backend() is released
            return "privateuseone"
    return "cpu"

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

def model_snapshot_folder(model, preferred_revision: str | None = None):
    """ Try to find the preferred revision, but fallback to another revision if necessary. """
    import diffusers
    storage_folder = os.path.join(diffusers.utils.DIFFUSERS_CACHE, model)
    if os.path.exists(os.path.join(storage_folder, 'model_index.json')): # converted model
        snapshot_folder = storage_folder
    else: # hub model
        revisions = {}
        for revision in os.listdir(os.path.join(storage_folder, "refs")):
            ref_path = os.path.join(storage_folder, "refs", revision)
            with open(ref_path) as f:
                commit_hash = f.read()

            snapshot_folder = os.path.join(storage_folder, "snapshots", commit_hash)
            if len(os.listdir(snapshot_folder)) > 1:
                revisions[revision] = snapshot_folder

        if len(revisions) == 0:
            return None
        elif preferred_revision in revisions:
            revision = preferred_revision
        elif preferred_revision in [None, "fp16"] and "main" in revisions:
            revision = "main"
        elif preferred_revision in [None, "main"] and "fp16" in revisions:
            revision = "fp16"
        else:
            revision = next(iter(revisions.keys()))
        snapshot_folder = revisions[revision]

    return snapshot_folder

def prompt_to_image(
    self,
    pipeline: Pipeline,
    
    model: str,

    scheduler: Scheduler,

    optimizations: Optimizations,

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

    # Stability SDK
    key: str | None = None,

    **kwargs
) -> Generator[ImageGenerationResult, None, None]:
    match pipeline:
        case Pipeline.STABLE_DIFFUSION:
            import diffusers
            import torch
            from PIL import Image, ImageOps

            # Mostly copied from `diffusers.StableDiffusionPipeline`, with slight modifications to yield the latents at each step.
            class GeneratorPipeline(diffusers.StableDiffusionPipeline):
                @torch.no_grad()
                def __call__(
                    self,
                    prompt: Union[str, List[str]],
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

                    # 1. Check inputs. Raise error if not correct
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

                    # 4. Prepare timesteps
                    self.scheduler.set_timesteps(num_inference_steps, device=device)
                    timesteps = self.scheduler.timesteps

                    # 5. Prepare latent variables
                    num_channels_latents = self.unet.in_channels
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

                    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
                    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

                    # 7. Denoising loop
                    for i, t in enumerate(self.progress_bar(timesteps)):
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
                        yield ImageGenerationResult.step_preview(self, kwargs['step_preview_mode'], width, height, latents, generator, i)

                    # 8. Post-processing
                    image = self.decode_latents(latents)

                    # TODO: Add UI to enable this.
                    # 9. Run safety checker
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
            pipe = load_pipe(self, "prompt", GeneratorPipeline, model, optimizations, scheduler, device)

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
            
            # Seamless
            _configure_model_padding(pipe.unet, seamless_axes)
            _configure_model_padding(pipe.vae, seamless_axes)

            # Inference
            with (torch.inference_mode() if device not in ('mps', "privateuseone") else nullcontext()), \
                (torch.autocast(device) if optimizations.can_use("amp", device) else nullcontext()):
                    yield from pipe(
                        prompt=prompt,
                        height=height,
                        width=width,
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

            answers = client.generate(
                prompt=prompt,
                width=width or 512,
                height=height or 512,
                cfg_scale=cfg_scale,
                sampler=scheduler.stability_sdk(),
                steps=steps,
                seed=seed
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

def _conv_forward_asymmetric(self, input, weight, bias):
    import torch.nn as nn
    """
    Patch for Conv2d._conv_forward that supports asymmetric padding
    """
    if input.device.type == "privateuseone":
        # DML pad() will wrongly fill the tensor in constant mode with the supplied value
        # (default 0) when padding on both ends of a dimension, can't split to two calls.
        working = nn.functional.pad(input, self._reversed_padding_repeated_twice, mode='circular')
        pad_w0, pad_w1, pad_h0, pad_h1 = self._reversed_padding_repeated_twice
        if self.asymmetric_padding_mode[0] == 'constant':
            working[:, :, :, :pad_w0] = 0
            if pad_w1 > 0:
                working[:, :, :, -pad_w1:] = 0
        if self.asymmetric_padding_mode[1] == 'constant':
            working[:, :, :pad_h0] = 0
            if pad_h1 > 0:
                working[:, :, -pad_h1:] = 0
    else:
        working = nn.functional.pad(input, self.asymmetric_padding[0], mode=self.asymmetric_padding_mode[0])
        working = nn.functional.pad(working, self.asymmetric_padding[1], mode=self.asymmetric_padding_mode[1])
    return nn.functional.conv2d(working, weight, bias, self.stride, nn.modules.utils._pair(0), self.dilation, self.groups)

def _configure_model_padding(model, seamless_axes):
    import torch.nn as nn
    """
    Modifies the 2D convolution layers to use a circular padding mode based on the `seamless` and `seamless_axes` options.
    """
    seamless_axes = SeamlessAxes(seamless_axes)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if seamless_axes.x or seamless_axes.y:
                m.asymmetric_padding_mode = (
                    'circular' if seamless_axes.x else 'constant',
                    'circular' if seamless_axes.y else 'constant'
                )
                m.asymmetric_padding = (
                    (m._reversed_padding_repeated_twice[0], m._reversed_padding_repeated_twice[1], 0, 0),
                    (0, 0, m._reversed_padding_repeated_twice[2], m._reversed_padding_repeated_twice[3])
                )
                m._conv_forward = _conv_forward_asymmetric.__get__(m, nn.Conv2d)
            else:
                m._conv_forward = nn.Conv2d._conv_forward.__get__(m, nn.Conv2d)
                if hasattr(m, 'asymmetric_padding_mode'):
                    del m.asymmetric_padding_mode
                if hasattr(m, 'asymmetric_padding'):
                    del m.asymmetric_padding
