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
        depth_map = depth.render_depth_map(context.depsgraph, collection=collection, invert=invert)
        context.update(depth_map)
        return {
            'Depth Map': depth_map,
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
        openpose_map = openpose.render_openpose_map(context.depsgraph, collection=collection)
        context.update(openpose_map)
        return {
            'OpenPose Map': openpose_map
        }