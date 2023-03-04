import bpy

class DreamTexturesNodeTree(bpy.types.NodeTree):
    bl_idname = "dream_textures.node_tree"
    bl_label = "Dream Textures Node Editor"
    bl_icon = 'NODETREE'

    @classmethod
    def poll(cls, context):
        return context.scene.render.engine == 'DREAM_TEXTURES'