import bpy
from bpy.types import Panel
from bpy_extras.io_utils import ImportHelper
from ...generator_process.registrar import BackendTarget
from ...pil_to_image import *
from ...prompt_engineering import *
from ...operators.upscale import Upscale
from ...absolute_path import REAL_ESRGAN_WEIGHTS_PATH
from ..space_types import SPACE_TYPES
import os

def upscaling_panels():
    for space_type in SPACE_TYPES:
        class UpscalingPanel(Panel):
            """Panel for AI Upscaling"""
            bl_label = "AI Upscaling"
            bl_category = "Dream"
            bl_idname = f"DREAM_PT_dream_upscaling_panel_{space_type}"
            bl_space_type = space_type
            bl_region_type = 'UI'
            bl_options = {'DEFAULT_CLOSED'}

            @classmethod
            def poll(cls, context):
                if not BackendTarget[context.scene.dream_textures_prompt.backend].upscaling():
                    return False
                if cls.bl_space_type == 'NODE_EDITOR':
                    return context.area.ui_type == "ShaderNodeTree" or context.area.ui_type == "CompositorNodeTree"
                else:
                    return True

            def draw(self, context):
                layout = self.layout
                layout.use_property_split = True
                
                layout.prop(context.scene, "dream_textures_upscale_outscale")
                layout.prop(context.scene, "dream_textures_upscale_full_precision")
                layout.prop(context.scene, "dream_textures_upscale_seamless")
                
                if not context.scene.dream_textures_upscale_full_precision:
                    box = layout.box()
                    box.label(text="Note: Some GPUs do not support mixed precision math", icon="ERROR")
                    box.label(text="If you encounter an error, enable full precision.")
                
                if context.scene.dream_textures_info != "":
                    layout.label(text=context.scene.dream_textures_info, icon="INFO")
                else:
                    layout.operator(Upscale.bl_idname, icon="FULLSCREEN_ENTER")
        
        UpscalingPanel.__name__ = f"DREAM_PT_dream_troubleshooting_panel_{space_type}"
        yield UpscalingPanel