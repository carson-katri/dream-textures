import bpy
import hashlib
import numpy as np
from typing import List

from .notify_result import NotifyResult
from ..pil_to_image import *
from ..prompt_engineering import *
from ..generator_process import Generator
from .. import api
import time

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

class DreamTexture(bpy.types.Operator):
    bl_idname = "shade.dream_texture"
    bl_label = "Dream Texture"
    bl_description = "Generate a texture with AI"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        try:
            context.scene.dream_textures_prompt.validate(context)
        except:
            return False
        return Generator.shared().can_use()

    def execute(self, context):
        screen = context.screen
        scene = context.scene
        prompt = scene.dream_textures_prompt
        backend: api.Backend = prompt.get_backend()

        history_template = {prop: getattr(context.scene.dream_textures_prompt, prop) for prop in context.scene.dream_textures_prompt.__annotations__.keys()}
        history_template["iterations"] = 1
        history_template["random_seed"] = False
        
        is_file_batch = context.scene.dream_textures_prompt.prompt_structure == file_batch_structure.id
        file_batch_lines = []
        if is_file_batch:
            context.scene.dream_textures_prompt.iterations = 1
            file_batch_lines = [line.body for line in context.scene.dream_textures_prompt_file.lines if len(line.body.strip()) > 0]
            history_template["prompt_structure"] = custom_structure.id

        node_tree = context.material.node_tree if hasattr(context, 'material') and hasattr(context.material, 'node_tree') else None
        node_tree_center = np.array(node_tree.view_center) if node_tree is not None else None
        
        context.scene.seamless_result.update_args(history_template, as_id=True)

        # Setup the progress indicator
        def step_progress_update(self, context):
            if hasattr(context.area, "regions"):
                for region in context.area.regions:
                    if region.type == "UI":
                        region.tag_redraw()
            return None
        bpy.types.Scene.dream_textures_progress = bpy.props.IntProperty(name="", default=0, min=0, max=prompt.steps, update=step_progress_update)
        scene.dream_textures_info = "Starting..."

        last_data_block = None
        execution_start = time.time()
        def step_callback(progress: api.GenerationResult):
            nonlocal last_data_block
            scene.dream_textures_last_execution_time = f"{time.time() - execution_start:.2f} seconds"
            scene.dream_textures_progress = scene.dream_textures_progress + 1
            last_data_block = bpy_image(f"Step {scene.dream_textures_progress}/{prompt.steps}", progress.image.shape[1], progress.image.shape[0], progress.image.ravel(), last_data_block)
            for area in screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    area.spaces.active.image = last_data_block

        def callback(result: List[api.GenerationResult] | Exception):
            if isinstance(result, Exception):
                scene.dream_textures_info = ""
                scene.dream_textures_progress = 0
                eval('bpy.ops.' + NotifyResult.bl_idname)('INVOKE_DEFAULT', exception=repr(result))
                raise result
            else:
                nonlocal last_data_block
                for i, generation in enumerate(result):
                    # Create a trimmed image name
                    prompt_string = context.scene.dream_textures_prompt.prompt_structure_token_subject
                    seed_str_length = len(str(generation.seed))
                    trim_aware_name = (prompt_string[:54 - seed_str_length] + '..') if len(prompt_string) > 54 else prompt_string
                    name_with_trimmed_prompt = f"{trim_aware_name} ({generation.seed})"
                    
                    # Create the image datablock
                    image = bpy_image(name_with_trimmed_prompt, generation.image.shape[1], generation.image.shape[0], generation.image.ravel(), last_data_block)
                    last_data_block = None

                    # Add Image Texture node
                    if node_tree is not None:
                        nodes = node_tree.nodes
                        texture_node = nodes.new("ShaderNodeTexImage")
                        texture_node.image = image
                        texture_node.location = node_tree_center + (i * 260, -i * 297)
                        nodes.active = texture_node

                    # Open the image in any active image editors
                    for area in screen.areas:
                        if area.type == 'IMAGE_EDITOR':
                            area.spaces.active.image = image
                    scene.dream_textures_prompt.seed = str(generation.seed) # update property in case seed was sourced randomly or from hash

                    # create a hash from the Blender image datablock to use as unique ID of said image and store it in the prompt history
                    # and as custom property of the image. Needs to be a string because the int from the hash function is too large
                    image_hash = hashlib.sha256((np.array(image.pixels) * 255).tobytes()).hexdigest()
                    image['dream_textures_hash'] = image_hash
                    scene.dream_textures_prompt.hash = image_hash
                    history_entry = context.scene.dream_textures_history.add()
                    for key, value in history_template.items():
                        match key:
                            case 'control_nets':
                                for net in value:
                                    n = history_entry.control_nets.add()
                                    for prop in n.__annotations__.keys():
                                        setattr(n, prop, getattr(net, prop))
                            case _:
                                setattr(history_entry, key, value)
                    history_entry.seed = str(generation.seed)
                    history_entry.hash = image_hash
                    if is_file_batch:
                        history_entry.prompt_structure_token_subject = file_batch_lines[i]
                scene.dream_textures_info = ""
                scene.dream_textures_progress = 0
        
        backend.generate(**prompt.generate_args(context), step_callback=step_callback, callback=callback)

        return {"FINISHED"}

def kill_generator(context=bpy.context):
    Generator.shared_close()
    try:
        context.scene.dream_textures_info = ""
        context.scene.dream_textures_progress = 0
    except:
        pass

class ReleaseGenerator(bpy.types.Operator):
    bl_idname = "shade.dream_textures_release_generator"
    bl_label = "Release Generator"
    bl_description = "Releases the generator class to free up VRAM"
    bl_options = {'REGISTER'}

    def execute(self, context):
        kill_generator(context)
        return {'FINISHED'}

class CancelGenerator(bpy.types.Operator):
    bl_idname = "shade.dream_textures_stop_generator"
    bl_label = "Cancel Generator"
    bl_description = "Stops the generator without reloading everything next time"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        gen = Generator.shared()
        return hasattr(gen, "_active_generation_future") and gen._active_generation_future is not None and not gen._active_generation_future.cancelled and not gen._active_generation_future.done

    def execute(self, context):
        gen = Generator.shared()
        gen._active_generation_future.cancel()
        context.scene.dream_textures_info = ""
        context.scene.dream_textures_progress = 0
        return {'FINISHED'}
