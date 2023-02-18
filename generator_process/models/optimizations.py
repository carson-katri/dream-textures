from dataclasses import dataclass
from typing import Annotated, Union, _AnnotatedAlias
import sys

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