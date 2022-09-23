from importlib.resources import path
import bpy
import asyncio
import os
import math

from ..preferences import StableDiffusionPreferences
from ..async_loop import *
from ..pil_to_image import *
from ..prompt_engineering import *
from ..absolute_path import WEIGHTS_PATH, absolute_path
from ..generator_process import GeneratorProcess
from .install_dependencies import are_dependencies_installed

import tempfile

# A shared `Generate` instance.
# This allows the slow model loading process to happen once,
# and re-use the model on subsequent calls.
generator = None

def image_has_alpha(img):
    b = 32 if img.is_float else 8
    return (
        img.depth == 2*b or   # Grayscale+Alpha
        img.depth == 4*b      # RGB+Alpha
    )

class DreamTexture(bpy.types.Operator):
    bl_idname = "shade.dream_texture"
    bl_label = "Dream Texture"
    bl_description = "Generate a texture with AI"
    bl_options = {'REGISTER'}
    
    def invoke(self, context, event):
        weights_installed = os.path.exists(WEIGHTS_PATH)
        if not weights_installed or not are_dependencies_installed():
            self.report({'ERROR'}, "Please complete setup in the preferences window.")
            return {"FINISHED"}
        else:
            return self.execute(context)

    async def dream_texture(self, context):
        history_entry = context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences.history.add()
        for prop in context.scene.dream_textures_prompt.__annotations__.keys():
            if hasattr(history_entry, prop):
                setattr(history_entry, prop, getattr(context.scene.dream_textures_prompt, prop))

        generated_prompt = context.scene.dream_textures_prompt.generate_prompt()

        node_tree = context.material.node_tree if hasattr(context, 'material') else None
        screen = context.screen
        last_data_block = None
        scene = context.scene

        global generator
        if generator is None:
            generator = GeneratorProcess()

        def step_progress_update(self, context):
            if hasattr(context.area, "regions"):
                for region in context.area.regions:
                    if region.type == "UI":
                        region.tag_redraw()
            return None

        bpy.types.Scene.dream_textures_progress = bpy.props.IntProperty(name="Progress", default=1, min=0, max=context.scene.dream_textures_prompt.steps + 1, update=step_progress_update)

        def image_writer(seed, width, height, pixels, upscaled=False):
            nonlocal last_data_block
            # Only use the non-upscaled texture, as upscaling is currently unsupported by the addon.
            if not upscaled:
                if last_data_block is not None:
                    bpy.data.images.remove(last_data_block)
                    last_data_block = None
                if generator is None or generator.process.poll() or width == 0 or height == 0:
                    return # process was closed
                image = bpy.data.images.new(f"{seed}", width=width, height=height)
                image.pixels[:] = pixels
                image.pack()
                if node_tree is not None:
                    nodes = node_tree.nodes
                    texture_node = nodes.new("ShaderNodeTexImage")
                    texture_node.image = image
                    nodes.active = texture_node
                for area in screen.areas:
                    if area.type == 'IMAGE_EDITOR':
                        area.spaces.active.image = image
                scene.dream_textures_progress = 0
                scene.dream_textures_prompt.seed = str(seed) # update property in case seed was sourced randomly or from hash
        
        def view_step(samples, step):
            step_progress(samples, step)
            nonlocal last_data_block
            for area in screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    step_image = pil_to_image(generator._sample_to_image(samples), name=f'Step {step + 1}/{scene.dream_textures_prompt.steps}')
                    area.spaces.active.image = step_image
                    if last_data_block is not None:
                        bpy.data.images.remove(last_data_block)
                    last_data_block = step_image
                    return # Only perform this on the first image editor found.
        
        def step_progress(samples, step):
            scene.dream_textures_progress = step + 1

        def save_temp_image(img, path=None):
            path = path if path is not None else tempfile.NamedTemporaryFile().name

            settings = scene.render.image_settings
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

        def perform():
            init_img = scene.init_img if scene.dream_textures_prompt.use_init_img else None
            if scene.dream_textures_prompt.use_inpainting:
                for area in screen.areas:
                    if area.type == 'IMAGE_EDITOR':
                        if area.spaces.active.image is not None and image_has_alpha(area.spaces.active.image):
                            init_img = area.spaces.active.image
            init_img_path = None
            if init_img is not None:
                init_img_path = save_temp_image(init_img)

            generator.prompt2image(
                # prompt string (no default)
                prompt=generated_prompt,
                # iterations (1); image count=iterations
                iterations=scene.dream_textures_prompt.iterations,
                # refinement steps per iteration
                steps=scene.dream_textures_prompt.steps,
                # seed for random number generator
                seed=scene.dream_textures_prompt.get_seed(),
                # width of image, in multiples of 64 (512)
                width=scene.dream_textures_prompt.width,
                # height of image, in multiples of 64 (512)
                height=scene.dream_textures_prompt.height,
                # how strongly the prompt influences the image (7.5) (must be >1)
                cfg_scale=scene.dream_textures_prompt.cfgscale,
                # path to an initial image - its dimensions override width and height
                init_img=init_img_path,

                # generate tileable/seamless textures
                seamless=scene.dream_textures_prompt.seamless,

                fit=scene.dream_textures_prompt.fit,
                # strength for noising/unnoising init_img. 0.0 preserves image exactly, 1.0 replaces it completely
                strength=scene.dream_textures_prompt.strength,
                # a function or method that will be called each step
                step_callback=view_step if scene.dream_textures_prompt.show_steps else step_progress,
                # a function or method that will be called each time an image is generated
                image_callback=image_writer,
                
                sampler_name=scene.dream_textures_prompt.sampler,

                full_precision=scene.dream_textures_prompt.full_precision
            )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, perform)

    def execute(self, context):
        async_task = asyncio.ensure_future(self.dream_texture(context))
        # async_task.add_done_callback(done_callback)
        ensure_async_loop()

        return {'FINISHED'}

class ReleaseGenerator(bpy.types.Operator):
    bl_idname = "shade.dream_textures_release_generator"
    bl_label = "Release Generator"
    bl_description = "Releases the generator class to free up VRAM"
    bl_options = {'REGISTER'}

    def execute(self, context):
        global generator
        if generator:
            generator.kill()
        generator = None
        context.scene.dream_textures_progress = 0
        return {'FINISHED'}