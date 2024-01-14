import bpy
from bpy.props import FloatProperty, EnumProperty, PointerProperty, IntProperty, BoolProperty

def lora_options(self, context):
    return [
        None if model is None else (model.id, model.name, model.description)
        for model in context.scene.dream_textures_prompt.get_backend().list_lora_models(context)
    ]

class Lora(bpy.types.PropertyGroup):
    lora: EnumProperty(name="LoRA", items=lora_options, description="Specify which LoRA to use")
    weight: FloatProperty(name="Weight", default=1.0, description="Increases the strength of the LoRA's effect")
    enabled: BoolProperty(name="Enabled", default=True)

class LorasAddMenu(bpy.types.Menu):
    bl_idname = "DREAM_MT_loras_add"
    bl_label = "Add LoRA"

    def draw(self, context):
        layout = self.layout

        for model in lora_options(self, context):
            if model is None:
                layout.separator()
            else:
                layout.operator("dream_textures.loras_add", text=model[1]).lora = model[0]

class LorasAdd(bpy.types.Operator):
    bl_idname = "dream_textures.loras_add"
    bl_label = "Add LoRA"

    lora: EnumProperty(name="LoRA", items=lora_options)

    def execute(self, context):
        net = context.scene.dream_textures_prompt.loras.add()
        net.lora = self.lora
        return {'FINISHED'}

class LorasRemove(bpy.types.Operator):
    bl_idname = "dream_textures.loras_remove"
    bl_label = "Remove LoRA"

    index: IntProperty(name="Index")

    def execute(self, context):
        context.scene.dream_textures_prompt.loras.remove(self.index)
        return {'FINISHED'}