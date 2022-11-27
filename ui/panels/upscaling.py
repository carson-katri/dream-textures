from bpy.types import Panel
from ...generator_process.registrar import BackendTarget
from ...pil_to_image import *
from ...prompt_engineering import *
from ...operators.upscale import Upscale
from .dream_texture import create_panel, advanced_panel
from ..space_types import SPACE_TYPES

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
                layout.use_property_decorate = False

                prompt = context.scene.dream_textures_upscale_prompt
                
                layout.prop(prompt, "prompt_structure_token_subject")
                layout.prop(context.scene, "dream_textures_upscale_tile_size")

                if context.scene.dream_textures_upscale_tile_size > 128:
                    warning_box = layout.box()
                    warning_box.label(text="Warning", icon="ERROR")
                    warning_box.label(text="Large tile sizes consume more VRAM.")

        UpscalingPanel.__name__ = UpscalingPanel.bl_idname
        class ActionsPanel(Panel):
            """Panel for AI Upscaling Actions"""
            bl_category = "Dream"
            bl_label = "Actions"
            bl_idname = f"DREAM_PT_dream_upscaling_actions_panel_{space_type}"
            bl_space_type = space_type
            bl_region_type = 'UI'
            bl_parent_id = UpscalingPanel.bl_idname
            bl_options = {'HIDE_HEADER'}

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
                
                if context.scene.dream_textures_info != "":
                    layout.label(text=context.scene.dream_textures_info, icon="INFO")
                else:
                    image = None
                    for area in context.screen.areas:
                        if area.type == 'IMAGE_EDITOR':
                            image = area.spaces.active.image
                    
                    row = layout.row()
                    row.scale_y = 1.5
                    row.operator(
                        Upscale.bl_idname,
                        text=f"Upscale to {image.size[0] * 4}x{image.size[1] * 4}" if image is not None else "Upscale",
                        icon="FULLSCREEN_ENTER"
                    )
        yield UpscalingPanel
        advanced_panels = [*create_panel(space_type, 'UI', UpscalingPanel.bl_idname, advanced_panel, lambda context: context.scene.dream_textures_upscale_prompt)]
        outer_panel = advanced_panels[0]
        outer_original_idname = outer_panel.bl_idname
        outer_panel.bl_idname += "_upscaling"
        for panel in advanced_panels:
            panel.bl_idname += "_upscaling"
            if panel.bl_parent_id == outer_original_idname:
                panel.bl_parent_id = outer_panel.bl_idname
            yield panel
        yield ActionsPanel