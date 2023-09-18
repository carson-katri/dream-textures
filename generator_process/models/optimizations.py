from enum import Enum
from typing import Annotated, Union, _AnnotatedAlias
import functools
import os
import sys
from dataclasses import dataclass

from .upscale_tiler import tiled_decode_latents


class CPUOffload(Enum):
    OFF = "off"
    MODEL = "model"
    SUBMODULE = "submodule"

    def __bool__(self):
        return self != CPUOffload.OFF


@dataclass(eq=True)
class Optimizations:
    attention_slicing: bool = True
    attention_slice_size: Union[str, int] = "auto"
    cudnn_benchmark: Annotated[bool, "cuda"] = False
    tf32: Annotated[bool, "cuda"] = False
    amp: Annotated[bool, "cuda"] = False
    half_precision: Annotated[bool, {"cuda", "dml"}] = True
    cpu_offload: Annotated[str, {"cuda", "dml"}] = CPUOffload.OFF
    channels_last_memory_format: bool = False
    sdp_attention: bool = True
    batch_size: int = 1
    vae_slicing: bool = True
    vae_tiling: str = "off"
    vae_tile_size: int = 512
    vae_tile_blend: int = 64
    cfg_end: float = 1.0

    cpu_only: bool = False

    @staticmethod
    def infer_device() -> str:
        from ...absolute_path import absolute_path
        if sys.platform == "darwin":
            return "mps"
        elif os.path.exists(absolute_path(".python_dependencies/torch_directml")):
            return "dml"
        else:
            return "cuda"

    @classmethod
    def device_supports(cls, property, device) -> bool:
        annotation = cls.__annotations__.get(property, None)
        if isinstance(annotation, _AnnotatedAlias):
            opt_dev = annotation.__metadata__[0]
            if isinstance(opt_dev, str):
                return opt_dev == device
            return device in opt_dev
        return annotation is not None

    def can_use(self, property, device) -> bool:
        return self.device_supports(property, device) and getattr(self, property)

    def can_use_half(self, device):
        if self.half_precision and device == "cuda":
            import torch
            name = torch.cuda.get_device_name()
            return not ("GTX 1650" in name or "GTX 1660" in name)
        return self.can_use("half_precision", device)

    def cpu_offloading(self, device):
        return self.cpu_offload if self.device_supports("cpu_offload", device) else CPUOffload.OFF
    
    def apply(self, pipeline, device):
        """
        Apply the optimizations to a diffusers pipeline.

        All exceptions are ignored to make this more general purpose across different pipelines.
        """
        import torch

        if not self.cpu_offloading(device):
            pipeline = pipeline.to(device)

        torch.backends.cudnn.benchmark = self.can_use("cudnn_benchmark", device)
        torch.backends.cuda.matmul.allow_tf32 = self.can_use("tf32", device)

        try:
            if self.can_use("sdp_attention", device):
                from diffusers.models.attention_processor import AttnProcessor2_0
                pipeline.unet.set_attn_processor(AttnProcessor2_0())
            elif self.can_use("attention_slicing", device):
                pipeline.enable_attention_slicing(self.attention_slice_size)
            else:
                pipeline.disable_attention_slicing()  # will also disable AttnProcessor2_0
        except: pass
        
        try:
            if pipeline.device != pipeline._execution_device:
                pass # pipeline is already offloaded, offloading again can cause `pipeline._execution_device` to be incorrect
            elif self.cpu_offloading(device) == CPUOffload.MODEL:
                # adapted from diffusers.StableDiffusionPipeline.enable_model_cpu_offload() to allow DirectML device and unimplemented pipelines
                from accelerate import cpu_offload_with_hook

                hook = None
                models = []
                # text_encoder can be None in SDXL Pipeline but not text_encoder_2
                if pipeline.text_encoder is not None:
                    models.append(pipeline.text_encoder)
                if hasattr(pipeline, "text_encoder_2"):
                    models.append(pipeline.text_encoder_2)
                models.extend([pipeline.unet, pipeline.vae])
                if hasattr(pipeline, "controlnet"):
                    models.append(pipeline.controlnet)
                for cpu_offloaded_model in models:
                    _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

                if getattr(pipeline, "safety_checker", None) is not None:
                    _, hook = cpu_offload_with_hook(pipeline.safety_checker, device, prev_module_hook=hook)

                # We'll offload the last model manually.
                pipeline.final_offload_hook = hook
            elif self.cpu_offloading(device) == CPUOffload.SUBMODULE:
                # adapted from diffusers.StableDiffusionPipeline.enable_sequential_cpu_offload() to allow DirectML device and unimplemented pipelines
                from accelerate import cpu_offload

                models = []
                # text_encoder can be None in SDXL Pipeline but not text_encoder_2
                if pipeline.text_encoder is not None:
                    models.append(pipeline.text_encoder)
                if hasattr(pipeline, "text_encoder_2"):
                    models.append(pipeline.text_encoder_2)
                models.extend([pipeline.unet, pipeline.vae])
                if hasattr(pipeline, "controlnet"):
                    models.append(pipeline.controlnet)
                for cpu_offloaded_model in models:
                    cpu_offload(cpu_offloaded_model, device)

                if getattr(pipeline, "safety_checker", None) is not None:
                    cpu_offload(pipeline.safety_checker, device, offload_buffers=True)
        except: pass
        
        try:
            if self.can_use("channels_last_memory_format", device):
                pipeline.unet.to(memory_format=torch.channels_last)
            else:
                pipeline.unet.to(memory_format=torch.contiguous_format)
        except: pass

        try:
            if self.can_use("vae_slicing", device):
                # Not many pipelines implement the enable_vae_slicing()/disable_vae_slicing()
                # methods but all they do is forward their call to the vae anyway.
                pipeline.vae.enable_slicing()
            else:
                pipeline.vae.disable_slicing()
        except: pass

        try:
            if self.vae_tiling != "off":
                if not isinstance(pipeline.vae.decode, functools.partial):
                    pipeline.vae.decode = functools.partial(tiled_decode_latents.__get__(pipeline), pre_patch=pipeline.vae.decode)
                pipeline.vae.decode.keywords['optimizations'] = self
            elif self.vae_tiling == "off" and isinstance(pipeline.vae.decode, functools.partial):
                pipeline.vae.decode = pipeline.vae.decode.keywords["pre_patch"]
        except: pass
        
        from .. import directml_patches
        if device == "dml":
            directml_patches.enable(pipeline)
        else:
            directml_patches.disable(pipeline)

        return pipeline