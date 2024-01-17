import bpy

class DreamTexturesNodeTree(bpy.types.NodeTree):
    bl_idname = "DreamTexturesNodeTree"
    bl_label = "Dream Textures Node Editor"
    bl_description = "Nodes for the Dream Textures Render Engine"
    bl_icon = 'NODETREE'

    @classmethod
    def poll(cls, context):
        return context.scene.render.engine == 'DREAM_TEXTURES'