import bpy
from .operators.dream_texture import DreamTexture
from .operators.open_latest_version import OpenLatestVersion, new_version_available

class ShaderMenu(bpy.types.Menu):
    bl_idname = "OBJECT_MT_dream_textures"
    bl_label = "Dream Textures"

    def draw(self, context):
        layout = self.layout

        layout.operator(DreamTexture.bl_idname)
        if new_version_available():
            layout.operator(OpenLatestVersion.bl_idname)

def shader_menu_draw(self, context):
    self.layout.menu(ShaderMenu.bl_idname)