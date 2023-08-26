import os

from ..models import ModelConfig


def convert_original_stable_diffusion_to_diffusers(
    self,
    checkpoint_path: str,
    model_config: ModelConfig,
) -> str:
    from diffusers.utils import DIFFUSERS_CACHE
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
    pipe = download_from_original_stable_diffusion_ckpt(
        checkpoint_path,
        original_config_file=model_config.original_config,
        from_safetensors=checkpoint_path.endswith(".safetensors"),
        pipeline_class=model_config.pipeline
    )
    dump_path = os.path.join(DIFFUSERS_CACHE, os.path.splitext(os.path.basename(checkpoint_path))[0])
    pipe.save_pretrained(dump_path)
    return dump_path
