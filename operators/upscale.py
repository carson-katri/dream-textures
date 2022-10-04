import bpy
import tempfile
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import sys
from ..generator_process import GeneratorProcess

upscale_options = [
    ("2", "2x", "", 2),
    ("4", "4x", "", 4),
]

generator_advance = None
timer = None

def remove_timer(context):
    global timer
    if timer:
        context.window_manager.event_timer_remove(timer)
        timer = None

class Upscale(bpy.types.Operator):
    bl_idname = "shade.dream_textures_upscale"
    bl_label = "Upscale"
    bl_description = ("Upscale with Real-ESRGAN")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        global timer
        return timer is None

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        try:
            next(generator_advance)
        except StopIteration:
            remove_timer(context)
            return {'FINISHED'}
        except Exception as e:
            remove_timer(context)
            raise e
        return {'RUNNING_MODAL'}

    def execute(self, context):
        scene = context.scene
        screen = context.screen
        node_tree = context.material.node_tree if hasattr(context, 'material') else None
        active_node = next((node for node in node_tree.nodes if node.select and node.bl_idname == 'ShaderNodeTexImage'), None)

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
            for area in context.screen.areas:
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

        generator = GeneratorProcess.shared()

        args = {
            'input': input_image_path,
            'name': input_image.name,
            'outscale': int(context.scene.dream_textures_upscale_outscale),
        }
        global generator_advance
        generator_advance = generator.upscale(args, image_callback, info_callback, exception_callback)
        context.window_manager.modal_handler_add(self)
        self.timer = context.window_manager.event_timer_add(1 / 15, window=context.window)

        return {"RUNNING_MODAL"}