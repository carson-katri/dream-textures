import bpy
from ..node import DreamTexturesNode
from ..annotations import depth
from ..annotations import normal
from ..annotations import openpose
from ..annotations import ade20k
from ..annotations import viewport
import numpy as np

annotation_src = (
    ('collection', 'Collection', 'Render the annotation for a specific collection'),
    ('scene', 'Scene', 'Render the annotation for the entire scene'),
)

def _update_annotation_inputs(self, context):
    inputs = {socket.name: socket for socket in self.inputs}
    inputs['Collection'].enabled = self.src == 'collection'

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

class NodeAnnotationNormal(DreamTexturesNode):
    bl_idname = "dream_textures.node_annotation_normal"
    bl_label = "Normal Map"

    src: bpy.props.EnumProperty(name="", items=annotation_src, update=_update_annotation_inputs)

    def init(self, context):
        self.inputs.new("NodeSocketCollection", "Collection")

        self.outputs.new("NodeSocketColor", "Normal Map")

    def draw_buttons(self, context, layout):
        layout.prop(self, "src")

    def execute(self, context, collection):
        normal_map = normal.render_normal_map(context.depsgraph, collection=collection if self.src == 'collection' else None)
        context.update(normal_map)
        return {
            'Normal Map': normal_map,
        }

class NodeAnnotationOpenPose(DreamTexturesNode):
    bl_idname = "dream_textures.node_annotation_openpose"
    bl_label = "OpenPose Map"

    src: bpy.props.EnumProperty(name="", items=annotation_src, update=_update_annotation_inputs)

    def init(self, context):
        self.inputs.new("NodeSocketCollection", "Collection")

        self.outputs.new("NodeSocketColor", "OpenPose Map")

    def draw_buttons(self, context, layout):
        layout.prop(self, "src")

    def execute(self, context, collection):
        openpose_map = openpose.render_openpose_map(context.depsgraph, collection=collection if self.src == 'collection' else None)
        context.update(openpose_map)
        return {
            'OpenPose Map': openpose_map
        }

class NodeAnnotationADE20K(DreamTexturesNode):
    bl_idname = "dream_textures.node_annotation_ade20k"
    bl_label = "ADE20K Segmentation Map"

    src: bpy.props.EnumProperty(name="", items=annotation_src, update=_update_annotation_inputs)

    def init(self, context):
        self.inputs.new("NodeSocketCollection", "Collection")

        self.outputs.new("NodeSocketColor", "Segmentation Map")

    def draw_buttons(self, context, layout):
        layout.prop(self, "src")

    def execute(self, context, collection):
        ade20k_map = ade20k.render_ade20k_map(context.depsgraph, collection=collection if self.src == 'collection' else None)
        context.update(ade20k_map)
        return {
            'Segmentation Map': ade20k_map
        }

class NodeAnnotationViewport(DreamTexturesNode):
    bl_idname = "dream_textures.node_annotation_viewport"
    bl_label = "Viewport Color"

    def init(self, context):
        self.outputs.new("NodeSocketColor", "Viewport Color")

    def draw_buttons(self, context, layout):
        pass

    def execute(self, context):
        color = viewport.render_viewport_color(context.depsgraph)
        context.update(color)
        return {
            'Viewport Color': color
        }