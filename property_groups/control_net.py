import bpy
from bpy.props import FloatProperty, EnumProperty, PointerProperty

def control_net_options(self, context):
    return [
        None if model is None else (model.id, model.name, model.description)
        for model in context.scene.dream_textures_prompt.get_backend().list_controlnet_models(context)
    ]

class ControlNet(bpy.types.PropertyGroup):
    control_net: EnumProperty(name="ControlNet", items=control_net_options, description="Specify which ControlNet to use")
    conditioning_scale: FloatProperty(name="ControlNet Conditioning Scale", default=1.0, description="Increases the strength of the ControlNet's effect")
    control_image: PointerProperty(type=bpy.types.Image)

class SCENE_UL_ControlNetList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        layout.separator()
        layout.prop(item, "control_net", text="")
        layout.prop(item, "conditioning_scale", text="")
        layout.template_ID(item, "control_image", open="image.open")

class ControlNetsAdd(bpy.types.Operator):
    bl_idname = "dream_textures.control_nets_add"
    bl_label = "Add ControlNet"

    def execute(self, context):
        context.scene.dream_textures_prompt.control_nets.add()
        return {'FINISHED'}
class ControlNetsRemove(bpy.types.Operator):
    bl_idname = "dream_textures.control_nets_remove"
    bl_label = "Add ControlNet"

    def execute(self, context):
        context.scene.dream_textures_prompt.control_nets.remove(context.scene.dream_textures_prompt.active_control_net)
        return {'FINISHED'}