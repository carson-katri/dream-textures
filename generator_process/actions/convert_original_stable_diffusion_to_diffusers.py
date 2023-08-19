import os
import enum
from ...absolute_path import absolute_path

class ModelConfig(enum.Enum):
    AUTO_DETECT = "auto-detect"
    STABLE_DIFFUSION_1 = "v1"
    STABLE_DIFFUSION_2_BASE = "v2 (512, epsilon)"
    STABLE_DIFFUSION_2 = "v2 (768, v_prediction)"
    STABLE_DIFFUSION_2_DEPTH = "v2 (depth)"
    STABLE_DIFFUSION_2_INPAINTING = "v2 (inpainting)"
    STABLE_DIFFUSION_XL_BASE = "XL (base)"
    STABLE_DIFFUSION_XL_REFINER = "XL (refiner)"

    @property
    def original_config(self):
        match self:
            case ModelConfig.AUTO_DETECT:
                return None
            case ModelConfig.STABLE_DIFFUSION_1:
                return absolute_path("sd_configs/v1-inference.yaml")
            case ModelConfig.STABLE_DIFFUSION_2_BASE:
                return absolute_path("sd_configs/v2-inference.yaml")
            case ModelConfig.STABLE_DIFFUSION_2:
                return absolute_path("sd_configs/v2-inference-v.yaml")
            case ModelConfig.STABLE_DIFFUSION_2_DEPTH:
                return absolute_path("sd_configs/v2-midas-inference.yaml")
            case ModelConfig.STABLE_DIFFUSION_2_INPAINTING:
                return absolute_path("sd_configs/v2-inpainting-inference.yaml")
            case ModelConfig.STABLE_DIFFUSION_XL_BASE:
                return absolute_path("sd_configs/sd_xl_base.yaml")
            case ModelConfig.STABLE_DIFFUSION_XL_REFINER:
                return absolute_path("sd_configs/sd_xl_refiner.yaml")

    @property
    def pipeline(self):
        # allows for saving with correct _class_name in model_index.json
        import diffusers
        match self:
            case ModelConfig.AUTO_DETECT:
                return None
            case ModelConfig.STABLE_DIFFUSION_2_DEPTH:
                return diffusers.StableDiffusionDepth2ImgPipeline
            case ModelConfig.STABLE_DIFFUSION_2_INPAINTING:
                return diffusers.StableDiffusionInpaintPipeline
            case ModelConfig.STABLE_DIFFUSION_XL_BASE:
                return diffusers.StableDiffusionXLPipeline
            case ModelConfig.STABLE_DIFFUSION_XL_REFINER:
                return diffusers.StableDiffusionXLImg2ImgPipeline
            case _:
                return diffusers.StableDiffusionPipeline


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
