import bpy
import gpu
from gpu_extras.batch import batch_for_shader
import numpy as np
from ..node import DreamTexturesNode
from ..annotations import openpose
from ..annotations import depth

class NodeString(DreamTexturesNode):
    bl_idname = "dream_textures.node_string"
    bl_label = "String"

    value: bpy.props.StringProperty(name="")

    def init(self, context):
        self.outputs.new("NodeSocketString", "String")

    def draw_buttons(self, context, layout):
        layout.prop(self, "value")

    def execute(self, context):
        return {
            'String': self.value
        }

class NodeInteger(DreamTexturesNode):
    bl_idname = "dream_textures.node_integer"
    bl_label = "Integer"

    value: bpy.props.IntProperty(name="")

    def init(self, context):
        self.outputs.new("NodeSocketInt", "Integer")

    def draw_buttons(self, context, layout):
        layout.prop(self, "value")

    def execute(self, context):
        return {
            'Integer': self.value
        }

class NodeCollection(DreamTexturesNode):
    bl_idname = "dream_textures.node_collection"
    bl_label = "Collection"

    value: bpy.props.PointerProperty(type=bpy.types.Collection, name="")

    def init(self, context):
        self.outputs.new("NodeSocketCollection", "Collection")

    def draw_buttons(self, context, layout):
        layout.prop(self, "value")

    def execute(self, context):
        return {
            'Collection': self.value
        }

class NodeImage(DreamTexturesNode):
    bl_idname = "dream_textures.node_image"
    bl_label = "Image"

    value: bpy.props.PointerProperty(type=bpy.types.Image)

    def init(self, context):
        self.outputs.new("NodeSocketColor", "Image")

    def draw_buttons(self, context, layout):
        layout.prop(self, "value", text="")

    def execute(self, context):
        return {
            'Image': np.array(self.value.pixels).reshape((*self.value.size, self.value.channels))
        }

class NodeRenderProperties(DreamTexturesNode):
    bl_idname = "dream_textures.node_render_properties"
    bl_label = "Render Properties"

    def init(self, context):
        self.outputs.new("NodeSocketInt", "Resolution X")
        self.outputs.new("NodeSocketInt", "Resolution Y")

    def draw_buttons(self, context, layout):
        pass

    def execute(self, context):
        return {
            'Resolution X': context.depsgraph.scene.render.resolution_x,
            'Resolution Y': context.depsgraph.scene.render.resolution_y,
        }