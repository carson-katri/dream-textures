import bpy
from bpy.types import Panel
from ..property_groups.dream_prompt import draw_dream_prompt_ui
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

    draw_dream_prompt_ui(context, layout, scene.dream_textures_prompt)
    
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