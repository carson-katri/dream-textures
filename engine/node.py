import bpy
from .node_tree import DreamTexturesNodeTree

class DreamTexturesNode(bpy.types.Node):
    @classmethod
    def poll(cls, tree):
        return tree.bl_idname == DreamTexturesNodeTree.bl_idname