import bpy
import numpy as np
from typing import List, Literal
from .. import api
from ..prompt_engineering import custom_structure
from ..generator_process import Generator
from .dream_texture import CancelGenerator
from .. import image_utils

upscale_options = [
    ("2", "2x", "", 2),
    ("4", "4x", "", 4),
    ("8", "8x", "", 8),
]

def get_source_image(context):
    node_tree = context.material.node_tree if hasattr(context, 'material') else None
    active_node = next((node for node in node_tree.nodes if node.select and node.bl_idname == 'ShaderNodeTexImage'), None) if node_tree is not None else None
    if active_node is not None and active_node.image is not None:
        return active_node.image
    elif context.area.type == 'IMAGE_EDITOR':
        return context.area.spaces.active.image
    else:
        input_image = None
        for area in context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                if area.spaces.active.image is not None:
                    input_image = area.spaces.active.image
        return input_image

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

        input_image = get_source_image(context)
        if input_image is None:
            self.report({"ERROR"}, "No open image in the Image Editor space, or selected Image Texture node.")
            return {"FINISHED"}
        image_pixels = image_utils.bpy_to_np(input_image)

        generated_args = context.scene.dream_textures_upscale_prompt.generate_args(context)
        context.scene.dream_textures_upscale_seamless_result.update_args(generated_args)

        # Setup the progress indicator
        def step_progress_update(self, context):
            if hasattr(context.area, "regions"):
                for region in context.area.regions:
                    if region.type == "UI":
                        region.tag_redraw()
            return None
        bpy.types.Scene.dream_textures_progress = bpy.props.IntProperty(name="", default=0, min=0, max=generated_args.steps, update=step_progress_update)
        scene.dream_textures_info = "Starting..."

        last_data_block = None
        def step_callback(progress: List[api.GenerationResult]) -> bool:
            nonlocal last_data_block
            if last_data_block is None:
                bpy.types.Scene.dream_textures_progress = bpy.props.IntProperty(name="", default=progress[-1].progress, min=0, max=progress[-1].total, update=step_progress_update)
            
            scene.dream_textures_progress = progress[-1].progress
            if progress[-1].image is not None:
                last_data_block = image_utils.np_to_bpy(progress[-1].image, f"Tile {progress[-1].progress}/{progress[-1].total}", last_data_block)
            for area in screen.areas:
                if area.type == 'IMAGE_EDITOR' and not area.spaces.active.use_image_pin:
                    area.spaces.active.image = last_data_block
            return CancelGenerator.should_continue

        def callback(results: List[api.GenerationResult] | Exception):
            if isinstance(results, Exception):
                scene.dream_textures_info = ""
                scene.dream_textures_progress = 0
                CancelGenerator.should_continue = None
            else:
                nonlocal last_data_block
                if last_data_block is not None:
                    bpy.data.images.remove(last_data_block)
                    last_data_block = None
                if results[-1].image is None:
                    return
                image = image_utils.np_to_bpy(results[-1].image, f"{input_image.name} (Upscaled)", last_data_block)
                for area in screen.areas:
                    if area.type == 'IMAGE_EDITOR' and not area.spaces.active.use_image_pin:
                        area.spaces.active.image = image
                if active_node is not None:
                    active_node.image = image
                scene.dream_textures_info = ""
                scene.dream_textures_progress = 0
                CancelGenerator.should_continue = None
        
        prompt = context.scene.dream_textures_upscale_prompt
        prompt.prompt_structure = custom_structure.id
        backend: api.Backend = prompt.get_backend()
        generated_args.task = api.models.task.Upscale(image=image_pixels, tile_size=context.scene.dream_textures_upscale_tile_size, blend=context.scene.dream_textures_upscale_blend)
        CancelGenerator.should_continue = True
        backend.generate(
            generated_args, step_callback=step_callback, callback=callback
        )
        
        return {"FINISHED"}