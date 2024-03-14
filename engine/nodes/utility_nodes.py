import bpy
import numpy as np
import random
from ..node import DreamTexturesNode
from ...property_groups.dream_prompt import seed_clamp
from ... import image_utils

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

class NodeFramePath(DreamTexturesNode):
    bl_idname = "dream_textures.node_frame_path"
    bl_label = "Frame Path"

    def init(self, context):
        self.inputs.new("NodeSocketInt", "Frame")

        self.outputs.new("NodeSocketString", "Frame Path")

    def draw_buttons(self, context, layout):
        pass

    def execute(self, context, frame):
        return {
            'Frame Path': context.depsgraph.scene.render.frame_path(frame=int(frame)),
        }

class NodeCropImage(DreamTexturesNode):
    bl_idname = "dream_textures.node_crop_image"
    bl_label = "Crop Image"

    def init(self, context):
        self.inputs.new("NodeSocketColor", "Image")
        self.inputs.new("NodeSocketInt", "X")
        self.inputs.new("NodeSocketInt", "Y")
        self.inputs.new("NodeSocketInt", "Width")
        self.inputs.new("NodeSocketInt", "Height")

        self.outputs.new("NodeSocketColor", "Cropped Image")

    def draw_buttons(self, context, layout):
        pass

    def execute(self, context, image, x, y, width, height):
        x, y = int(x), int(y)
        width, height = int(width), int(height)
        result = image[y:y+height, x:x+width, ...]
        context.update(result)
        return {
            'Cropped Image': result,
        }

class NodeResizeImage(DreamTexturesNode):
    bl_idname = "dream_textures.node_resize_image"
    bl_label = "Resize Image"

    def init(self, context):
        self.inputs.new("NodeSocketColor", "Image")
        self.inputs.new("NodeSocketInt", "Width")
        self.inputs.new("NodeSocketInt", "Height")

        self.outputs.new("NodeSocketColor", "Resized Image")

    def draw_buttons(self, context, layout):
        pass

    def execute(self, context, image, width, height):
        result = image_utils.resize(image, (width, height))
        context.update(result)
        return {
            'Resized Image': result,
        }

class NodeJoinImages(DreamTexturesNode):
    bl_idname = "dream_textures.node_join_images"
    bl_label = "Join Images"

    direction: bpy.props.EnumProperty(name="", items=(
        ('horizontal', 'Horizontal', ''),
        ('vertical', 'Vertical', ''),
    ))

    def init(self, context):
        self.inputs.new("NodeSocketColor", "A")
        self.inputs.new("NodeSocketColor", "B")

        self.outputs.new("NodeSocketColor", "Joined Images")

    def draw_buttons(self, context, layout):
        layout.prop(self, "direction")

    def execute(self, context, a, b):
        match self.direction:
            case 'horizontal':
                result = np.hstack([a, b])
            case 'vertical':
                result = np.vstack([a, b])
        context.update(result)
        return {
            'Joined Images': result,
        }

class NodeSeparateColor(DreamTexturesNode):
    bl_idname = "dream_textures.node_separate_color"
    bl_label = "Separate Color"

    def init(self, context):
        self.inputs.new("NodeSocketColor", "Color")

        self.outputs.new("NodeSocketFloat", "Red")
        self.outputs.new("NodeSocketFloat", "Green")
        self.outputs.new("NodeSocketFloat", "Blue")
        self.outputs.new("NodeSocketFloat", "Alpha")

    def draw_buttons(self, context, layout):
        pass

    def execute(self, context, color):
        return {
            'Red': color[..., 0],
            'Green': color[..., 1] if color.shape[-1] > 1 else 0,
            'Blue': color[..., 2] if color.shape[-1] > 2 else 0,
            'Alpha': color[..., 3] if color.shape[-1] > 3 else 0,
        }

class NodeCombineColor(DreamTexturesNode):
    bl_idname = "dream_textures.node_combine_color"
    bl_label = "Combine Color"

    def init(self, context):
        self.inputs.new("NodeSocketFloat", "Red")
        self.inputs.new("NodeSocketFloat", "Green")
        self.inputs.new("NodeSocketFloat", "Blue")
        self.inputs.new("NodeSocketFloat", "Alpha")

        self.outputs.new("NodeSocketColor", "Color")

    def draw_buttons(self, context, layout):
        pass

    def execute(self, context, red, green, blue, alpha):
        return {
            'Color': np.stack([red, green, blue, alpha], axis=-1)
        }

class NodeColorCorrect(DreamTexturesNode):
    bl_idname = "dream_textures.node_color_correct"
    bl_label = "Color Correct"

    mode: bpy.props.EnumProperty(name="Mode", items=(
        ('histogram', 'Match Histograms', ''),
    ))

    def init(self, context):
        self.inputs.new("NodeSocketColor", "Image")
        self.inputs.new("NodeSocketColor", "Target")

        self.outputs.new("NodeSocketColor", "Image")

    def draw_buttons(self, context, layout):
        layout.prop(self, "mode", text="")

    def execute(self, context, image, target):
        match self.mode:
            case 'histogram':
                flat_image = image.ravel()
                flat_target = target.ravel()

                _, image_indices, image_counts = np.unique(flat_image, return_inverse=True, return_counts=True)
                target_values, target_counts = np.unique(flat_target, return_counts=True)
                
                image_quantiles = np.cumsum(image_counts).astype(np.float64)
                image_quantiles /= image_quantiles[-1]
                
                target_quantiles = np.cumsum(target_counts).astype(np.float64)
                target_quantiles /= target_quantiles[-1]

                result = np.interp(image_quantiles, target_quantiles, target_values)[image_indices].reshape(image.shape)
        return {
            'Image': result
        }

class NodeSwitch(DreamTexturesNode):
    bl_idname = "dream_textures.node_switch"
    bl_label = "Switch"

    def init(self, context):
        self.inputs.new("NodeSocketBool", "Switch")
        self.inputs.new("NodeSocketColor", "False")
        self.inputs.new("NodeSocketColor", "True")

        self.outputs.new("NodeSocketColor", "Output")

    def draw_buttons(self, context, layout):
        pass

    def execute(self, context, switch, false, true):
        return {
            'Output': true() if switch else false()
        }

class NodeCompare(DreamTexturesNode):
    bl_idname = "dream_textures.node_compare"
    bl_label = "Compare"

    operation: bpy.props.EnumProperty(name="", items=(
        ('<', 'Less Than', ''),
        ('<=', 'Less Than or Equal', ''),
        ('>', 'Greater Than', ''),
        ('>=', 'Greater Than or Equal', ''),
        ('==', 'Equal', ''),
        ('!=', 'Not Equal', ''),
    ))

    def init(self, context):
        self.inputs.new("NodeSocketFloat", "A")
        self.inputs.new("NodeSocketFloat", "B")

        self.outputs.new("NodeSocketBool", "Result")

    def draw_buttons(self, context, layout):
        layout.prop(self, "operation")

    def execute(self, context, a, b):
        match self.operation:
            case '<':
                result = a < b
            case '<=':
                result = a <= b
            case '>':
                result = a > b
            case '>=':
                result = a >= b
            case '==':
                result = a == b
            case '!=':
                result = a != b
        return {
            'Result': result
        }

class NodeReplaceString(DreamTexturesNode):
    bl_idname = "dream_textures.node_replace_string"
    bl_label = "Replace String"

    def init(self, context):
        self.inputs.new("NodeSocketString", "String")
        self.inputs.new("NodeSocketString", "Find")
        self.inputs.new("NodeSocketString", "Replace")

        self.outputs.new("NodeSocketString", "String")

    def draw_buttons(self, context, layout):
        pass

    def execute(self, context, string, find, replace):
        return {
            'String': string.replace(find, replace)
        }