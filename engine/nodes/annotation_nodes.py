import bpy
from ..node import DreamTexturesNode
from ..annotations import depth
from ..annotations import openpose

annotation_src = (
    ('collection', 'Collection', 'Render the annotation for a specific collection'),
    ('scene', 'Scene', 'Render the annotation for the entire scene'),
)

def _update_annotation_inputs(self, context):
    self.inputs['Collection'].enabled = self.src == 'collection'

class NodeAnnotationDepth(DreamTexturesNode):
    bl_idname = "dream_textures.node_annotation_depth"
    bl_label = "Depth Map"

    src: bpy.props.EnumProperty(name="", items=annotation_src, update=_update_annotation_inputs)

    def init(self, context):
        self.inputs.new("NodeSocketCollection", "Collection")
        self.inputs.new("NodeSocketBool", "Invert")

        self.outputs.new("NodeSocketColor", "Depth Map")

    def draw_buttons(self, context, layout):
        layout.prop(self, "src")

    def execute(self, context, collection, invert):
        depth_map = depth.render_depth_map(context.depsgraph, collection=collection if self.src == 'collection' else None, invert=invert)
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