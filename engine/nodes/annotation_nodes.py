import bpy
from ..node import DreamTexturesNode
from ..annotations import depth
from ..annotations import openpose

class NodeAnnotationDepth(DreamTexturesNode):
    bl_idname = "dream_textures.node_annotation_depth"
    bl_label = "Depth Map"

    def init(self, context):
        self.inputs.new("NodeSocketCollection", "Collection")
        self.inputs.new("NodeSocketBool", "Invert")

        self.outputs.new("NodeSocketColor", "Depth Map")

    def draw_buttons(self, context, layout):
        pass

    def execute(self, context, collection, invert):
        return {
            'Depth Map': depth.render_depth_map(context, collection=collection, invert=invert),
        }

class NodeAnnotationOpenPose(DreamTexturesNode):
    bl_idname = "dream_textures.node_annotation_openpose"
    bl_label = "OpenPose Map"

    def init(self, context):
        self.inputs.new("NodeSocketCollection", "Collection")

        self.outputs.new("NodeSocketColor", "OpenPose Map")

    def draw_buttons(self, context, layout):
        pass

    def execute(self, context, collection):
        return {
            'OpenPose Map': openpose.render_openpose_map(context, collection=collection)
        }