import bpy

class ObjectPrompt(bpy.types.PropertyGroup):
    bl_label = "ObjectPrompt"
    bl_idname = "dream_textures.ObjectPrompt"

    prompt: bpy.props.StringProperty(name="Prompt")