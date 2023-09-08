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
    CONTROL_NET_1_5 = "1.5 (ControlNet)"
    CONTROL_NET_2_1 = "2.1 (ControlNet)"

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
            case ModelConfig.CONTROL_NET_1_5:
                return absolute_path("sd_configs/cldm_v15.yaml")
            case ModelConfig.CONTROL_NET_2_1:
                return absolute_path("sd_configs/cldm_v21.yaml")

    @property
    def pipeline(self):
        # allows for saving with correct _class_name in model_index.json and necessary for some models to import
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
            case ModelConfig.CONTROL_NET_1_5 | ModelConfig.CONTROL_NET_2_1:
                return diffusers.ControlNetModel
            case _:
                return diffusers.StableDiffusionPipeline
