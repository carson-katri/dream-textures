import bpy
from .operators.dream_texture import DreamTexture
from .operators.open_latest_version import OpenLatestVersion, new_version_available
from .operators.help_panel import HelpPanel

class ShaderMenu(bpy.types.Menu):
    bl_idname = "OBJECT_MT_dream_textures"
    bl_label = "Dream Textures"

    def draw(self, context):
        layout = self.layout

        layout.operator(DreamTexture.bl_idname, icon="IMAGE")
        layout.operator(HelpPanel.bl_idname, icon="QUESTION")
        if new_version_available():
            layout.operator(OpenLatestVersion.bl_idname, icon="IMPORT")

def shader_menu_draw(self, context):
    self.layout.menu(ShaderMenu.bl_idname)

def image_menu_draw(self, context):
    self.layout.menu(ShaderMenu.bl_idname)