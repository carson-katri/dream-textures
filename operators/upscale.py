import bpy
import tempfile
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import sys
from ..generator_process import Generator

class Upscale(bpy.types.Operator):
    bl_idname = "shade.dream_textures_upscale"
    bl_label = "Upscale"
    bl_description = ("Upscale with Real-ESRGAN")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return Generator.shared().can_use()

    def execute(self, context):
        scene = context.scene
        screen = context.screen
        node_tree = context.material.node_tree if hasattr(context, 'material') else None
        active_node = next((node for node in node_tree.nodes if node.select and node.bl_idname == 'ShaderNodeTexImage'), None) if node_tree is not None else None

        def step_progress_update(self, context):
            if hasattr(context.area, "regions"):
                for region in context.area.regions:
                    if region.type == "UI":
                        region.tag_redraw()
            return None

        bpy.types.Scene.dream_textures_info = bpy.props.StringProperty(name="Info", update=step_progress_update)

        def save_temp_image(img, path=None):
            path = path if path is not None else tempfile.NamedTemporaryFile().name

            settings = context.scene.render.image_settings
            file_format = settings.file_format
            mode = settings.color_mode
            depth = settings.color_depth

            settings.file_format = 'PNG'
            settings.color_mode = 'RGBA'
            settings.color_depth = '8'

            img.save_render(path)

            settings.file_format = file_format
            settings.color_mode = mode
            settings.color_depth = depth

            return path

        input_image = None
        input_image_path = None
        if active_node is not None and active_node.image is not None:
            input_image = active_node.image
            input_image_path = save_temp_image(input_image)
        else:
            for area in context.screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    if area.spaces.active.image is not None:
                        input_image = area.spaces.active.image
                        input_image_path = save_temp_image(input_image)
        
        if input_image is None:
            self.report({"ERROR"}, "No open image in the Image Editor space, or selected Image Texture node.")
            return {"FINISHED"}

        def bpy_image(name, width, height, pixels):
            image = bpy.data.images.new(name, width=width, height=height)
            image.pixels[:] = pixels
            image.pack()
            return image

        def image_callback(shared_memory_name, seed, width, height):
            scene.dream_textures_info = ""
            shared_memory = SharedMemory(shared_memory_name)
            image = bpy_image(seed + ' (Upscaled)', width, height, np.frombuffer(shared_memory.buf,dtype=np.float32))
            for area in screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    area.spaces.active.image = image
            if active_node is not None:
                active_node.image = image
            shared_memory.close()

        def info_callback(msg=""):
            scene.dream_textures_info = msg
        def exception_callback(fatal, msg, trace):
            scene.dream_textures_info = ""
            self.report({'ERROR'}, msg)
            if trace:
                print(trace, file=sys.stderr)

        # args = {
        #     'input': input_image_path,
        #     'name': input_image.name,
        #     'outscale': int(context.scene.dream_textures_upscale_outscale),
        #     'full_precision': context.scene.dream_textures_upscale_full_precision,
        #     'seamless': context.scene.dream_textures_upscale_seamless
        # }

        def image_done(future):
            image = future.result()
            image = bpy_image("diffusers-upscaled", image.shape[0], image.shape[1], image.ravel())
            for area in screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    area.spaces.active.image = image
            if active_node is not None:
                active_node.image = image
        Generator.shared().upscale(input_image_path, "brick wall", context.scene.dream_textures_upscale_full_precision).add_done_callback(image_done)
        
        return {"FINISHED"}