import bpy
import numpy as np
import random
from ..node import DreamTexturesNode
from ...property_groups.dream_prompt import seed_clamp

class NodeMath(DreamTexturesNode):
    bl_idname = "dream_textures.node_math"
    bl_label = "Math"

    operation: bpy.props.EnumProperty(
        name="Operation",
        items=(
            ("add", "Add", ""),
            ("subtract", "Subtract", ""),
            ("multiply", "Multiply", ""),
            ("divide", "Divide", ""),
        )
    )

    def init(self, context):
        self.inputs.new("NodeSocketFloat", "A")
        self.inputs.new("NodeSocketFloat", "B")

        self.outputs.new("NodeSocketFloat", "Value")

    def draw_buttons(self, context, layout):
        layout.prop(self, "operation", text="")

    def perform(self, a, b):
        match self.operation:
            case 'add':
                return a + b
            case 'subtract':
                return a - b
            case 'multiply':
                return a * b
            case 'divide':
                return a / b

    def execute(self, context, a, b):
        return {
            'Value': self.perform(a, b)
        }

class NodeRandomValue(DreamTexturesNode):
    bl_idname = "dream_textures.node_random_value"
    bl_label = "Random Value"

    data_type: bpy.props.EnumProperty(name="", items=(
        ('integer', 'Integer', '', 1),
    ))

    def init(self, context):
        self.inputs.new("NodeSocketInt", "Min")
        self.inputs.new("NodeSocketInt", "Max").default_value = np.iinfo(np.int32).max
        
        self.outputs.new("NodeSocketInt", "Value")

    def draw_buttons(self, context, layout):
        layout.prop(self, "data_type")

    def execute(self, context, min, max):
        return {
            'Value': random.randrange(min, max)
        }

class NodeRandomSeed(DreamTexturesNode):
    bl_idname = "dream_textures.node_random_seed"
    bl_label = "Random Seed"

    def init(self, context):
        self.outputs.new("NodeSocketInt", "Value")

    def draw_buttons(self, context, layout):
        pass

    def execute(self, context):
        return {
            'Value': random.randrange(0, np.iinfo(np.uint32).max)
        }

class NodeSeed(DreamTexturesNode):
    bl_idname = "dream_textures.node_seed"
    bl_label = "Seed"

    seed: bpy.props.StringProperty(name="", default="", update=seed_clamp)

    def init(self, context):
        self.outputs.new("NodeSocketInt", "Value")

    def draw_buttons(self, context, layout):
        layout.prop(self, "seed")

    def execute(self, context):
        return {
            'Value': int(self.seed)
        }

class NodeClamp(DreamTexturesNode):
    bl_idname = "dream_textures.node_clamp"
    bl_label = "Clamp"

    def init(self, context):
        self.inputs.new("NodeSocketFloat", "Value")
        self.inputs.new("NodeSocketFloat", "Min")
        self.inputs.new("NodeSocketFloat", "Max")

        self.outputs.new("NodeSocketFloat", "Result")

    def draw_buttons(self, context, layout):
        pass

    def execute(self, context, value, min, max):
        return {
            'Result': np.clip(value, min, max)
        }