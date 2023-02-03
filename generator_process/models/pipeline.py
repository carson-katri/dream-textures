import enum
import os

class Pipeline(enum.IntEnum):
    STABLE_DIFFUSION = 0

    STABILITY_SDK = 1

    @staticmethod
    def local_available():
        from ...absolute_path import absolute_path
        return os.path.exists(absolute_path(".python_dependencies/diffusers"))

    @staticmethod
    def directml_available():
        from ...absolute_path import absolute_path
        return os.path.exists(absolute_path(".python_dependencies/torch_directml"))

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