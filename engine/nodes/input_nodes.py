import bpy
import gpu
from gpu_extras.batch import batch_for_shader
import numpy as np
from ..node import DreamTexturesNode
from ..annotations import openpose
from ..annotations import depth
from ... import image_utils

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
        layout.template_ID(self, "value", open="image.open")
        if self.value is not None:
            layout.prop(self.value.colorspace_settings, "name", text="Color Space")

    def execute(self, context):
        result = image_utils.bpy_to_np(self.value, color_space="Linear", top_to_bottom=False)
        context.update(result)
        return {
            'Image': result
        }

class NodeImageFile(DreamTexturesNode):
    bl_idname = "dream_textures.node_image_file"
    bl_label = "Image File"

    def init(self, context):
        self.inputs.new("NodeSocketString", "Path")

        self.outputs.new("NodeSocketColor", "Image")

    def draw_buttons(self, context, layout):
        pass

    def execute(self, context, path):
        pixels = image_utils.image_to_np(path, default_color_space="sRGB", to_color_space="Linear", top_to_bottom=False)
        context.update(pixels)
        return {
            'Image': pixels
        }

class NodeRenderProperties(DreamTexturesNode):
    bl_idname = "dream_textures.node_render_properties"
    bl_label = "Render Properties"

    def init(self, context):
        self.outputs.new("NodeSocketInt", "Resolution X")
        self.outputs.new("NodeSocketInt", "Resolution Y")
        self.outputs.new("NodeSocketString", "Output Filepath")
        self.outputs.new("NodeSocketInt", "Frame")

    def draw_buttons(self, context, layout):
        pass

    def execute(self, context):
        return {
            'Resolution X': context.depsgraph.scene.render.resolution_x,
            'Resolution Y': context.depsgraph.scene.render.resolution_y,
            'Output Filepath': context.depsgraph.scene.render.filepath,
            'Frame': context.depsgraph.scene.frame_current
        }