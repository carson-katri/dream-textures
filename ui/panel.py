import bpy
from bpy.types import Panel
from ..pil_to_image import *
from ..prompt_engineering import *
from ..operators.dream_texture import DreamTexture, ReleaseGenerator
from ..operators.view_history import SCENE_UL_HistoryList, RecallHistoryEntry
from ..operators.open_latest_version import OpenLatestVersion, is_force_show_download, new_version_available
from ..help_section import help_section
from ..preferences import StableDiffusionPreferences
import sys

SPACE_TYPES = {'IMAGE_EDITOR', 'NODE_EDITOR'}

def draw_panel(self, context):
    layout = self.layout
    scene = context.scene

    if is_force_show_download():
        layout.operator(OpenLatestVersion.bl_idname, icon="IMPORT", text="Download Latest Release")
    elif new_version_available():
        layout.operator(OpenLatestVersion.bl_idname, icon="IMPORT")

    prompt_box = layout.box()
    prompt_box_heading = prompt_box.row()
    prompt_box_heading.label(text="Prompt")
    prompt_box_heading.prop(scene.dream_textures_prompt, "prompt_structure")
    structure = next(x for x in prompt_structures if x.id == scene.dream_textures_prompt.prompt_structure)
    for segment in structure.structure:
        segment_row = prompt_box.row()
        enum_prop = 'prompt_structure_token_' + segment.id + '_enum'
        is_custom = getattr(scene.dream_textures_prompt, enum_prop) == 'custom'
        if is_custom:
            segment_row.prop(scene.dream_textures_prompt, 'prompt_structure_token_' + segment.id)
        segment_row.prop(scene.dream_textures_prompt, enum_prop, icon_only=is_custom)
    
    size_box = layout.box()
    size_box.label(text="Configuration")
    size_box.prop(scene.dream_textures_prompt, "width")
    size_box.prop(scene.dream_textures_prompt, "height")
    size_box.prop(scene.dream_textures_prompt, "seamless")
    
    if not scene.dream_textures_prompt.use_init_img:
        for area in context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                if area.spaces.active.image is not None:
                    inpainting_box = layout.box()
                    inpainting_heading = inpainting_box.row()
                    inpainting_heading.prop(scene.dream_textures_prompt, "use_inpainting")
                    inpainting_heading.label(text="Inpaint Open Image")
                    break

    if not scene.dream_textures_prompt.use_inpainting or area.spaces.active.image is None:
        init_img_box = layout.box()
        init_img_heading = init_img_box.row()
        init_img_heading.prop(scene.dream_textures_prompt, "use_init_img")
        init_img_heading.label(text="Init Image")
        if scene.dream_textures_prompt.use_init_img:
            init_img_box.template_ID(context.scene, "init_img", open="image.open")
            init_img_box.prop(scene.dream_textures_prompt, "strength")
            init_img_box.prop(scene.dream_textures_prompt, "fit")

    advanced_box = layout.box()
    advanced_box_heading = advanced_box.row()
    advanced_box_heading.prop(scene.dream_textures_prompt, "show_advanced", icon="DOWNARROW_HLT" if scene.dream_textures_prompt.show_advanced else "RIGHTARROW_THIN", emboss=False, icon_only=True)
    advanced_box_heading.label(text="Advanced Configuration")
    if scene.dream_textures_prompt.show_advanced:
        if sys.platform not in {'darwin'}:
            advanced_box.prop(scene.dream_textures_prompt, "full_precision")
        advanced_box.prop(scene.dream_textures_prompt, "random_seed")
        seed_row = advanced_box.row()
        seed_row.prop(scene.dream_textures_prompt, "seed")
        seed_row.enabled = not scene.dream_textures_prompt.random_seed
        # advanced_box.prop(self, "iterations") # Disabled until supported by the addon.
        advanced_box.prop(scene.dream_textures_prompt, "steps")
        advanced_box.prop(scene.dream_textures_prompt, "cfg_scale")
        advanced_box.prop(scene.dream_textures_prompt, "sampler")
        advanced_box.prop(scene.dream_textures_prompt, "show_steps")
    
    row = layout.row()
    row.scale_y = 1.5
    if context.scene.dream_textures_progress <= 0:
        if context.scene.dream_textures_info != "":
            row.label(text=context.scene.dream_textures_info, icon="INFO")
        else:
            row.operator(DreamTexture.bl_idname, icon="PLAY", text="Generate")
    else:
        disabled_row = row.row()
        disabled_row.prop(context.scene, 'dream_textures_progress', slider=True)
        disabled_row.enabled = False
    row.operator(ReleaseGenerator.bl_idname, icon="X", text="")

def panels():
    for space_type in SPACE_TYPES:
        class DreamTexturePanel(Panel):
            """Creates a Panel in the scene context of the properties editor"""
            bl_label = "Dream Texture"
            bl_category = "Dream"
            bl_idname = f"DREAM_PT_dream_panel_{space_type}"
            bl_space_type = space_type
            bl_region_type = 'UI'

            @classmethod
            def poll(self, context):
                if self.bl_space_type == 'NODE_EDITOR':
                    return context.area.ui_type == "ShaderNodeTree" or context.area.ui_type == "CompositorNodeTree"
                else:
                    return True

            def draw(self, context):
                draw_panel(self, context)
        DreamTexturePanel.__name__ = f"DREAM_PT_dream_panel_{space_type}"
        yield DreamTexturePanel

def history_panels():
    for space_type in SPACE_TYPES:
        class HistoryPanel(Panel):
            """Panel for Dream Textures History"""
            bl_label = "History"
            bl_category = "Dream"
            bl_idname = f"DREAM_PT_dream_history_panel_{space_type}"
            bl_space_type = space_type
            bl_region_type = 'UI'

            selection: bpy.props.IntProperty(name="Selection")

            @classmethod
            def poll(self, context):
                if self.bl_space_type == 'NODE_EDITOR':
                    return context.area.ui_type == "ShaderNodeTree" or context.area.ui_type == "CompositorNodeTree"
                else:
                    return True

            def draw(self, context):
                if len(context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences.history) < 1:
                    header = context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences.history.add()
                else:
                    header = context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences.history[0]
                header.prompt_structure_token_subject = "SCENE_UL_HistoryList_header"
                self.layout.template_list("SCENE_UL_HistoryList", "", context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences, "history", context.scene, "dream_textures_history_selection")
                self.layout.operator(RecallHistoryEntry.bl_idname)
        HistoryPanel.__name__ = f"DREAM_PT_dream_history_panel_{space_type}"
        yield HistoryPanel

def troubleshooting_panels():
    for space_type in SPACE_TYPES:
        class TroubleshootingPanel(Panel):
            """Panel for Dream Textures Troubleshooting"""
            bl_label = "Troubleshooting"
            bl_category = "Dream"
            bl_idname = f"DREAM_PT_dream_troubleshooting_panel_{space_type}"
            bl_space_type = space_type
            bl_region_type = 'UI'
            bl_options = {'DEFAULT_CLOSED'}

            @classmethod
            def poll(self, context):
                if self.bl_space_type == 'NODE_EDITOR':
                    return context.area.ui_type == "ShaderNodeTree" or context.area.ui_type == "CompositorNodeTree"
                else:
                    return True

            def draw(self, context):
                help_section(self.layout, context)
        TroubleshootingPanel.__name__ = f"DREAM_PT_dream_troubleshooting_panel_{space_type}"
        yield TroubleshootingPanel