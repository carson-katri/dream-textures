import bpy
import nodeitems_utils
from ..node import DreamTexturesNode
import random

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

class NodeRandomValue(DreamTexturesNode):
    bl_idname = "dream_textures.node_random_value"
    bl_label = "Random Value"

    data_type: bpy.props.EnumProperty(name="", items=(
        ('integer', 'Integer', '', 1),
    ))

    def init(self, context):
        self.inputs.new("NodeSocketInt", "Min")
        self.inputs.new("NodeSocketInt", "Max")
        
        self.outputs.new("NodeSocketInt", "Value")

    def draw_buttons(self, context, layout):
        layout.prop(self, "data_type")

    def execute(self, context, min, max):
        return {
            'Value': random.randrange(min, max)
        }