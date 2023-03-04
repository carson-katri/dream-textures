import bpy
from .engine import DreamTexturesRenderEngine

class DreamTexturesNodeTree(bpy.types.NodeTree):
    bl_label = "Dream Textures Node Editor"
    bl_icon = 'NODETREE'

    @classmethod
    def poll(cls, context):
        return context.scene.render.engine == DreamTexturesRenderEngine.bl_idname