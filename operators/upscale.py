import bpy
import tempfile
from ..prompt_engineering import custom_structure
from ..generator_process import Generator

upscale_options = [
    ("2", "2x", "", 2),
    ("4", "4x", "", 4),
    ("8", "8x", "", 8),
]

class Upscale(bpy.types.Operator):
    bl_idname = "shade.dream_textures_upscale"
    bl_label = "Upscale"
    bl_description = ("Upscale with Real-ESRGAN")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return Generator.shared().can_use()

    def execute(self, context):
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

        def on_tile_complete(_, tile):
            image = bpy_image("diffusers-upscaled", tile.shape[0], tile.shape[1], tile.ravel())
            for area in screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    area.spaces.active.image = image

        def image_done(future):
            image = future.result()[-1]
            image = bpy_image("diffusers-upscaled", image.shape[0], image.shape[1], image.ravel())
            for area in screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    area.spaces.active.image = image
            if active_node is not None:
                active_node.image = image
        gen = Generator.shared()
        context.scene.dream_textures_upscale_prompt.prompt_structure = custom_structure.id
        f = gen.upscale(
            image=input_image_path,
            tile_size=context.scene.dream_textures_upscale_tile_size,
            **context.scene.dream_textures_upscale_prompt.generate_args()
        )
        f.add_response_callback(on_tile_complete)
        f.add_done_callback(image_done)
        gen._active_generation_future = f
        
        return {"FINISHED"}