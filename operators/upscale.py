import bpy
import numpy as np
from ..prompt_engineering import custom_structure
from ..generator_process import Generator
from ..generator_process.actions.upscale import ImageUpscaleResult

upscale_options = [
    ("2", "2x", "", 2),
    ("4", "4x", "", 4),
    ("8", "8x", "", 8),
]

def bpy_image(name, width, height, pixels, existing_image):
    if existing_image is not None and (existing_image.size[0] != width or existing_image.size[1] != height):
        bpy.data.images.remove(existing_image)
        existing_image = None
    if existing_image is None:
        image = bpy.data.images.new(name, width=width, height=height)
    else:
        image = existing_image
        image.name = name
    image.pixels.foreach_set(pixels)
    image.pack()
    image.update()
    return image

class Upscale(bpy.types.Operator):
    bl_idname = "shade.dream_textures_upscale"
    bl_label = "Upscale"
    bl_description = ("Upscale with Stable Diffusion x4 Upscaler")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return Generator.shared().can_use()

    def execute(self, context):
        screen = context.screen
        scene = context.scene
        node_tree = context.material.node_tree if hasattr(context, 'material') else None
        active_node = next((node for node in node_tree.nodes if node.select and node.bl_idname == 'ShaderNodeTexImage'), None) if node_tree is not None else None

        def step_progress_update(self, context):
            if hasattr(context.area, "regions"):
                for region in context.area.regions:
                    if region.type == "UI":
                        region.tag_redraw()
            return None

        bpy.types.Scene.dream_textures_info = bpy.props.StringProperty(name="Info", update=step_progress_update)

        input_image = None
        if active_node is not None and active_node.image is not None:
            input_image = active_node.image
        else:
            for area in context.screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    if area.spaces.active.image is not None:
                        input_image = area.spaces.active.image
        if input_image is None:
            self.report({"ERROR"}, "No open image in the Image Editor space, or selected Image Texture node.")
            return {"FINISHED"}
        image_pixels = np.flipud(
            (np.array(input_image.pixels) * 255)
                .astype(np.uint8)
                .reshape((input_image.size[1], input_image.size[0], input_image.channels))
        )

        generated_args = context.scene.dream_textures_upscale_prompt.generate_args()
        context.scene.dream_textures_upscale_seamless_result.update_args(generated_args)

        # Setup the progress indicator
        def step_progress_update(self, context):
            if hasattr(context.area, "regions"):
                for region in context.area.regions:
                    if region.type == "UI":
                        region.tag_redraw()
            return None
        bpy.types.Scene.dream_textures_progress = bpy.props.IntProperty(name="", default=0, min=0, max=generated_args['steps'], update=step_progress_update)
        scene.dream_textures_info = "Starting..."

        last_data_block = None
        def on_tile_complete(_, tile: ImageUpscaleResult):
            nonlocal last_data_block
            if last_data_block is None:
                bpy.types.Scene.dream_textures_progress = bpy.props.IntProperty(name="", default=tile.tile, min=0, max=tile.total, update=step_progress_update)
            if tile.final or tile.image is None:
                return
            
            scene.dream_textures_progress = tile.tile
            last_data_block = bpy_image(f"Tile {tile.tile}/{tile.total}", tile.image.shape[1], tile.image.shape[0], tile.image.ravel(), last_data_block)
            for area in screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    area.spaces.active.image = last_data_block

        def image_done(future):
            nonlocal last_data_block
            if last_data_block is not None:
                bpy.data.images.remove(last_data_block)
                last_data_block = None
            tile: ImageUpscaleResult = future.result(last_only=True)
            if tile.image is None:
                return
            image = bpy_image(f"{input_image.name} (Upscaled)", tile.image.shape[1], tile.image.shape[0], tile.image.ravel(), last_data_block)
            for area in screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    area.spaces.active.image = image
            if active_node is not None:
                active_node.image = image
            scene.dream_textures_info = ""
            scene.dream_textures_progress = 0
        gen = Generator.shared()
        context.scene.dream_textures_upscale_prompt.prompt_structure = custom_structure.id
        f = gen.upscale(
            image=image_pixels,
            tile_size=context.scene.dream_textures_upscale_tile_size,
            blend=context.scene.dream_textures_upscale_blend,
            **generated_args
        )
        f.add_response_callback(on_tile_complete)
        f.add_done_callback(image_done)
        gen._active_generation_future = f
        
        return {"FINISHED"}