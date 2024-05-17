import bpy
from bpy.props import FloatProperty, EnumProperty, PointerProperty, IntProperty, BoolProperty

from .. import api, image_utils
from ..diffusers_backend import DiffusersBackend
from ..generator_process import Generator
from ..generator_process.models.optimizations import Optimizations

def control_net_options(self, context):
    return [
        None if model is None else (model.id, model.name, model.description)
        for model in context.scene.dream_textures_prompt.get_backend().list_controlnet_models(context)
    ]

PROCESSOR_IDS = [
    ("none", "None", "No pre-processing"),
    None,
    ("depth_leres", "Depth (LeRes)", ""),
    ("depth_leres++", "Depth (LeRes++)", ""),
    ("depth_midas", "Depth (MiDaS)", ""),
    ("depth_zoe", "Depth (Zoe)", ""),
    None,
    ("canny", "Canny", "Canny edge detection"),
    ("mlsd", "M-LSD", ""),
    ("softedge_hed", "Soft Edge (HED)", ""),
    ("softedge_hedsafe", "Soft Edge (HED-Safe)", ""),
    ("softedge_pidinet", "Soft Edge (PidiNet)", ""),
    ("softedge_pidsafe", "Soft Edge (Pidsafe)", ""),
    None,
    ("lineart_anime", "Lineart (Anime)", ""),
    ("lineart_coarse", "Lineart (Coarse)", ""),
    ("lineart_realistic", "Lineart (Realistic)", ""),
    None,
    ("normal_bae", "Normal (BAE)", ""),
    None,
    ("openpose", "OpenPose", ""),
    ("openpose_face", "OpenPose (Face)", ""),
    ("openpose_faceonly", "OpenPose (Face Only)", ""),
    ("openpose_full", "OpenPose (Full)", ""),
    ("openpose_hand", "OpenPose (Hand)", ""),
    # ("dwpose", "DWPose", ""), # requires additional dependencies
    # ("mediapipe_face", "MediaPipe Face", ""), # requires additional dependencies
    None,
    ("scribble_hed", "Scribble (HED)", ""),
    ("scribble_pidinet", "Scribble (PidiNet)", ""),
    None,
    ("shuffle", "Shuffle", ""),
]

class ControlNet(bpy.types.PropertyGroup):
    control_net: EnumProperty(name="ControlNet", items=control_net_options, description="Specify which ControlNet to use")
    conditioning_scale: FloatProperty(name="Conditioning Scale", default=1.0, description="Increases the strength of the ControlNet's effect")
    control_image: PointerProperty(type=bpy.types.Image)
    processor_id: EnumProperty(
        name="Processor",
        items=PROCESSOR_IDS,
        description="Pre-process the control image"
    )
    enabled: BoolProperty(name="Enabled", default=True)

class ControlNetsAddMenu(bpy.types.Menu):
    bl_idname = "DREAM_MT_control_nets_add"
    bl_label = "Add ControlNet"

    def draw(self, context):
        layout = self.layout

        for model in control_net_options(self, context):
            if model is None:
                layout.separator()
            else:
                layout.operator("dream_textures.control_nets_add", text=model[1]).control_net = model[0]

class ControlNetsAdd(bpy.types.Operator):
    bl_idname = "dream_textures.control_nets_add"
    bl_label = "Add ControlNet"

    control_net: EnumProperty(name="ControlNet", items=control_net_options)

    def execute(self, context):
        net = context.scene.dream_textures_prompt.control_nets.add()
        net.control_net = self.control_net
        return {'FINISHED'}
class ControlNetsRemove(bpy.types.Operator):
    bl_idname = "dream_textures.control_nets_remove"
    bl_label = "Remove ControlNet"

    index: IntProperty(name="Index")

    def execute(self, context):
        context.scene.dream_textures_prompt.control_nets.remove(self.index)
        return {'FINISHED'}

class BakeControlNetImage(bpy.types.Operator):
    bl_idname = "dream_textures.control_net_bake"
    bl_label = "Bake Control Image"
    bl_description = "Runs the selected processor, and bakes the result to an image datablock"

    index: IntProperty(name="Index")

    def execute(self, context):
        prompt = context.scene.dream_textures_prompt

        net = prompt.control_nets[self.index]

        gen = Generator.shared()
        backend: api.Backend = prompt.get_backend()
        optimizations = backend.optimizations() if isinstance(backend, DiffusersBackend) else Optimizations()

        future = gen.controlnet_aux(
            processor_id=net.processor_id,
            image=image_utils.bpy_to_np(net.control_image, color_space=None),
            optimizations=optimizations
        )
        
        control_image = image_utils.np_to_bpy(
            future.result(last_only=True),
            f"{net.control_image.name} ({next(processor[1] for processor in PROCESSOR_IDS if processor != None and processor[0] == net.processor_id)})",
        )
        
        net.control_image = control_image
        net.processor_id = "none"

        for area in context.screen.areas:
            if area.type == 'IMAGE_EDITOR' and not area.spaces.active.use_image_pin:
                area.spaces.active.image = control_image

        return {'FINISHED'}