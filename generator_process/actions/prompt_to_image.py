from typing import Annotated, Union, _AnnotatedAlias, Generator, Callable, List, Optional
import enum
import os
from dataclasses import dataclass
from contextlib import nullcontext

from numpy.typing import NDArray
import numpy as np
import random
from .detect_seamless import SeamlessAxes


class Pipeline(enum.IntEnum):
    STABLE_DIFFUSION = 0

    STABILITY_SDK = 1

    @staticmethod
    def local_available():
        from ...absolute_path import absolute_path
        return os.path.exists(absolute_path(".python_dependencies/diffusers"))

    def __str__(self):
        return self.name
    
    def model(self):
        return True

    def init_img_actions(self):
        match self:
            case Pipeline.STABLE_DIFFUSION:
                return ['modify', 'inpaint', 'outpaint']
            case Pipeline.STABILITY_SDK:
                return ['modify', 'inpaint']
    
    def inpaint_mask_sources(self):
        match self:
            case Pipeline.STABLE_DIFFUSION:
                return ['alpha', 'prompt']
            case Pipeline.STABILITY_SDK:
                return ['alpha']
    
    def color_correction(self):
        match self:
            case Pipeline.STABLE_DIFFUSION:
                return True
            case Pipeline.STABILITY_SDK:
                return False
    
    def negative_prompts(self):
        match self:
            case Pipeline.STABLE_DIFFUSION:
                return True
            case Pipeline.STABILITY_SDK:
                return False
    
    def seamless(self):
        match self:
            case Pipeline.STABLE_DIFFUSION:
                return True
            case Pipeline.STABILITY_SDK:
                return False
    
    def upscaling(self):
        match self:
            case Pipeline.STABLE_DIFFUSION:
                return True
            case Pipeline.STABILITY_SDK:
                return False
    
    def depth(self):
        match self:
            case Pipeline.STABLE_DIFFUSION:
                return True
            case Pipeline.STABILITY_SDK:
                return False

class Scheduler(enum.Enum):
    LMS_DISCRETE = "LMS Discrete"
    DDIM = "DDIM"
    PNDM = "PNDM"
    DDPM = "DDPM"
    DPM_SOLVER_MULTISTEP = "DPM Solver Multistep"
    EULER_DISCRETE = "Euler Discrete"
    EULER_ANCESTRAL_DISCRETE = "Euler Ancestral Discrete"

    def create(self, pipeline, pretrained):
        import diffusers
        def scheduler_class():
            match self:
                case Scheduler.LMS_DISCRETE:
                    return diffusers.LMSDiscreteScheduler
                case Scheduler.DDIM:
                    return diffusers.DDIMScheduler
                case Scheduler.PNDM:
                    return diffusers.PNDMScheduler
                case Scheduler.DDPM:
                    return diffusers.DDPMScheduler
                case Scheduler.DPM_SOLVER_MULTISTEP:
                    return diffusers.DPMSolverMultistepScheduler
                case Scheduler.EULER_DISCRETE:
                    return diffusers.EulerDiscreteScheduler
                case Scheduler.EULER_ANCESTRAL_DISCRETE:
                    return diffusers.EulerAncestralDiscreteScheduler
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
    half_precision: Annotated[bool, "cuda"] = True
    sequential_cpu_offload: bool = False
    channels_last_memory_format: bool = False
    # xformers_attention: bool = False # FIXME: xFormers is not yet available.

    cpu_only: bool = False

    def can_use(self, property, device) -> bool:
        if not getattr(self, property):
            return False
        if isinstance(self.__annotations__.get(property, None), _AnnotatedAlias):
            annotation: _AnnotatedAlias = self.__annotations__[property]
            return annotation.__metadata__[0] == device
        return True
    
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
            if self.can_use("sequential_cpu_offload", device):
                pipeline.enable_sequential_cpu_offload()
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
        return pipeline

class StepPreviewMode(enum.Enum):
    NONE = "None"
    FAST = "Fast"
    ACCURATE = "Accurate"

@dataclass
class ImageGenerationResult:
    image: NDArray | None
    seed: int
    step: int
    final: bool

def choose_device(self) -> str:
    """
    Automatically select which PyTorch device to use.
    """
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
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

def prompt_to_image(
    self,
    pipeline: Pipeline,
    
    model: str,

    scheduler: Scheduler,

    optimizations: Optimizations,

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
                    generator: Optional[torch.Generator] = None,
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

                    # 8. Post-processing
                    image = self.decode_latents(latents)

                    # TODO: Add UI to enable this.
                    # 9. Run safety checker
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
            if hasattr(self, "_cached_pipe") and self._cached_pipe[1] == model and use_cpu_offload == self._cached_pipe[2]:
                pipe = self._cached_pipe[0]
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
                setattr(self, "_cached_pipe", (pipe, model, use_cpu_offload, snapshot_folder))

            # Scheduler
            is_stable_diffusion_2 = 'stabilityai--stable-diffusion-2' in model
            pipe.scheduler = scheduler.create(pipe, {
                'model_path': self._cached_pipe[3],
                'subfolder': 'scheduler',
            } if is_stable_diffusion_2 else None)

            # Optimizations
            pipe = optimizations.apply(pipe, device)

            # RNG
            generator = torch.Generator(device="cpu" if device == "mps" else device) # MPS does not support the `Generator` API
            if seed is None:
                seed = random.randrange(0, np.iinfo(np.uint32).max)
            generator = generator.manual_seed(seed)
            
            # Seamless
            _configure_model_padding(pipe.unet, seamless_axes)
            _configure_model_padding(pipe.vae, seamless_axes)

            # Inference
            with (torch.inference_mode() if device != 'mps' else nullcontext()), \
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
                width=width,
                height=height,
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
                            np.asarray(ImageOps.flip(image).convert('RGBA'), dtype=np.float32) / 255.,
                            seed,
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
