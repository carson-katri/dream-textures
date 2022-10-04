from bpy.types import Panel
from ...pil_to_image import *
from ...prompt_engineering import *
from ...help_section import help_section
from ..space_types import SPACE_TYPES

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