import bpy
from bpy.types import Panel
from ...pil_to_image import *
from ...prompt_engineering import *
from ...operators.dream_texture import DreamTexture, ReleaseGenerator
from ...operators.view_history import ExportHistorySelection, ImportPromptFile, RecallHistoryEntry, ClearHistory, RemoveHistorySelection
from ...operators.open_latest_version import OpenLatestVersion, is_force_show_download, new_version_available
from ...preferences import StableDiffusionPreferences
from ..space_types import SPACE_TYPES

def history_panels():
    for space_type in SPACE_TYPES:
        class HistoryPanel(Panel):
            """Panel for Dream Textures History"""
            bl_label = "History"
            bl_category = "Dream"
            bl_idname = f"DREAM_PT_dream_history_panel_{space_type}"
            bl_space_type = space_type
            bl_region_type = 'UI'

            @classmethod
            def poll(cls, context):
                if cls.bl_space_type == 'NODE_EDITOR':
                    return context.area.ui_type == "ShaderNodeTree" or context.area.ui_type == "CompositorNodeTree"
                else:
                    return True

            def draw(self, context):
                self.layout.template_list("SCENE_UL_HistoryList", "", context.scene, "dream_textures_history", context.scene, "dream_textures_history_selection")
                
                row = self.layout.row()
                row.prop(context.scene, "dream_textures_history_selection_preview")
                row.operator(RemoveHistorySelection.bl_idname, text="", icon="X")
                row.operator(ExportHistorySelection.bl_idname, text="", icon="EXPORT")

                self.layout.operator(RecallHistoryEntry.bl_idname)
                self.layout.operator(ClearHistory.bl_idname)
        HistoryPanel.__name__ = f"DREAM_PT_dream_history_panel_{space_type}"
        yield HistoryPanel