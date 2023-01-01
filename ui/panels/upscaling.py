from bpy.types import Panel
from ...pil_to_image import *
from ...prompt_engineering import *
from ...operators.upscale import Upscale
from ...operators.dream_texture import CancelGenerator, ReleaseGenerator
from ...generator_process.actions.detect_seamless import SeamlessAxes
from ...generator_process.actions.prompt_to_image import Pipeline
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
                if not Pipeline[context.scene.dream_textures_prompt.pipeline].upscaling():
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
                layout.prop(context.scene, "dream_textures_upscale_blend")

                layout.prop(prompt, "seamless_axes")

                if prompt.seamless_axes == SeamlessAxes.AUTO:
                    node_tree = context.material.node_tree if hasattr(context, 'material') else None
                    active_node = next((node for node in node_tree.nodes if node.select and node.bl_idname == 'ShaderNodeTexImage'), None) if node_tree is not None else None
                    init_image = None
                    if active_node is not None and active_node.image is not None:
                        init_image = active_node.image
                    else:
                        for area in context.screen.areas:
                            if area.type == 'IMAGE_EDITOR':
                                if area.spaces.active.image is not None:
                                    init_image = area.spaces.active.image
                    context.scene.dream_textures_upscale_seamless_result.check(init_image)
                    auto_row = layout.row()
                    auto_row.enabled = False
                    auto_row.prop(context.scene.dream_textures_upscale_seamless_result, "result")

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
                if not Pipeline[context.scene.dream_textures_prompt.pipeline].upscaling():
                    return False
                if cls.bl_space_type == 'NODE_EDITOR':
                    return context.area.ui_type == "ShaderNodeTree" or context.area.ui_type == "CompositorNodeTree"
                else:
                    return True

            def draw(self, context):
                layout = self.layout
                layout.use_property_split = True
                layout.use_property_decorate = False
                
                image = None
                for area in context.screen.areas:
                    if area.type == 'IMAGE_EDITOR':
                        image = area.spaces.active.image
                row = layout.row()
                row.scale_y = 1.5
                if context.scene.dream_textures_progress <= 0:
                    if context.scene.dream_textures_info != "":
                        row.label(text=context.scene.dream_textures_info, icon="INFO")
                    else:
                        row.operator(
                            Upscale.bl_idname,
                            text=f"Upscale to {image.size[0] * 4}x{image.size[1] * 4}" if image is not None else "Upscale",
                            icon="FULLSCREEN_ENTER"
                        )
                else:
                    disabled_row = row.row()
                    disabled_row.use_property_split = True
                    disabled_row.prop(context.scene, 'dream_textures_progress', slider=True)
                    disabled_row.enabled = False
                if CancelGenerator.poll(context):
                    row.operator(CancelGenerator.bl_idname, icon="CANCEL", text="")
                row.operator(ReleaseGenerator.bl_idname, icon="X", text="")
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