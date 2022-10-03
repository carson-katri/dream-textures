import bpy
import tempfile
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
    bl_label = f"Upscale"
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
        for area in context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                if area.spaces.active.image is not None:
                    input_image = save_temp_image(area.spaces.active.image)
        
        if input_image is None:
            self.report({"ERROR"}, "No open image in the Image Editor space")
            return {"FINISHED"}

        def image_callback(output_path):
            print("Received image callback")
            print(output_path)
            image = bpy.data.images.load(output_path)
            print("Saved")
            image.pack()
            print("Packed")

        def info_callback(msg=""):
            print("Info", msg)
        def exception_callback(fatal, msg, trace):
            print("Exception Callback", fatal, msg, trace)

        generator = GeneratorProcess.shared()

        args = {
            'input': input_image,
            'level': int(context.scene.dream_textures_upscale_target_size),
            'strength': float(context.scene.dream_textures_upscale_strength),
        }
        print("Running", args)
        global generator_advance
        generator_advance = generator.upscale(args, image_callback, info_callback, exception_callback)
        context.window_manager.modal_handler_add(self)
        self.timer = context.window_manager.event_timer_add(1 / 15, window=context.window)

        return {"RUNNING_MODAL"}