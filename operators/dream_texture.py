import bpy
import hashlib
import numpy as np
from typing import List, Literal

from .notify_result import NotifyResult
from ..prompt_engineering import *
from ..generator_process import Generator
from .. import api
from .. import image_utils
from ..generator_process.models.optimizations import Optimizations
from ..diffusers_backend import DiffusersBackend
import time
import math

def get_source_image(context, source: Literal['file', 'open_editor']):
    match source:
        case 'file':
            return context.scene.init_img
        case 'open_editor':
            if context.area.type == 'IMAGE_EDITOR':
                return context.area.spaces.active.image
            else:
                init_image = None
                for area in context.screen.areas:
                    if area.type == 'IMAGE_EDITOR':
                        if area.spaces.active.image is not None:
                            init_image = area.spaces.active.image
                return init_image
        case _:
            raise ValueError(f"unsupported source {repr(source)}")

class DreamTexture(bpy.types.Operator):
    bl_idname = "shade.dream_texture"
    bl_label = "Dream Texture"
    bl_description = "Generate a texture with AI"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        try:
            prompt = context.scene.dream_textures_prompt
            backend: api.Backend = prompt.get_backend()
            backend.validate(prompt.generate_args(context))
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
        node_tree_top_left = np.array(context.region.view2d.region_to_view(0, context.region.height)) if node_tree is not None else None
        screen = context.screen
        scene = context.scene

        generated_args = scene.dream_textures_prompt.generate_args(context)
        context.scene.seamless_result.update_args(generated_args)
        context.scene.seamless_result.update_args(history_template, as_id=True)

        def execute_backend(control_images):
            # Setup the progress indicator
            bpy.types.Scene.dream_textures_progress = bpy.props.IntProperty(name="", default=0, min=0, max=generated_args.steps)
            scene.dream_textures_info = "Starting..."

            # Get any init images
            try:
                init_image = get_source_image(context, prompt.init_img_src)
            except ValueError:
                init_image = None
            if init_image is not None:
                init_image_color_space = "sRGB"
                if scene.dream_textures_prompt.use_init_img and scene.dream_textures_prompt.modify_action_source_type in ['depth_map', 'depth']:
                    init_image_color_space = None
                init_image = image_utils.bpy_to_np(init_image, color_space=init_image_color_space)

            # Callbacks
            last_data_block = None
            execution_start = time.time()
            def step_callback(progress: List[api.GenerationResult]) -> bool:
                nonlocal last_data_block
                scene.dream_textures_last_execution_time = f"{time.time() - execution_start:.2f} seconds"
                scene.dream_textures_progress = progress[-1].progress
                for area in context.screen.areas:
                    for region in area.regions:
                        if region.type == "UI":
                            region.tag_redraw()
                image = api.GenerationResult.tile_images(progress)
                if image is None:
                    return CancelGenerator.should_continue
                last_data_block = image_utils.np_to_bpy(image, f"Step {progress[-1].progress}/{progress[-1].total}", last_data_block)
                for area in screen.areas:
                    if area.type == 'IMAGE_EDITOR' and not area.spaces.active.use_image_pin:
                        area.spaces.active.image = last_data_block
                return CancelGenerator.should_continue

            iteration = 0
            iteration_limit = len(file_batch_lines) if is_file_batch else generated_args.iterations
            iteration_square = math.ceil(math.sqrt(iteration_limit))
            node_pad = np.array((20, 20))
            node_size = np.array((240, 277)) + node_pad
            if node_tree is not None:
                # keep image nodes grid centered but don't go beyond top and left sides of nodes editor
                node_anchor = node_tree_center + node_size * 0.5 * (-iteration_square, (iteration_limit-1) // iteration_square + 1)
                node_anchor = np.array((np.maximum(node_tree_top_left[0], node_anchor[0]), np.minimum(node_tree_top_left[1], node_anchor[1]))) + node_pad * (0.5, -0.5)
            
            def callback(results: List[api.GenerationResult] | Exception):
                if isinstance(results, Exception):
                    scene.dream_textures_info = ""
                    scene.dream_textures_progress = 0
                    CancelGenerator.should_continue = None
                    if not isinstance(results, InterruptedError): # this is a user-initiated cancellation
                        eval('bpy.ops.' + NotifyResult.bl_idname)('INVOKE_DEFAULT', exception=repr(results))
                    raise results
                else:
                    nonlocal last_data_block
                    nonlocal iteration
                    for result in results:
                        if result.image is None or result.seed is None:
                            continue
                        
                        # Create a trimmed image name
                        prompt_string = context.scene.dream_textures_prompt.prompt_structure_token_subject
                        seed_str_length = len(str(result.seed))
                        trim_aware_name = (prompt_string[:54 - seed_str_length] + '..') if len(prompt_string) > 54 else prompt_string
                        name_with_trimmed_prompt = f"{trim_aware_name} ({result.seed})"
                        image = image_utils.np_to_bpy(result.image, name_with_trimmed_prompt, last_data_block)
                        last_data_block = None
                        if node_tree is not None:
                            nodes = node_tree.nodes
                            texture_node = nodes.new("ShaderNodeTexImage")
                            texture_node.image = image
                            texture_node.location = node_anchor + node_size * ((iteration % iteration_square), -(iteration // iteration_square))
                            nodes.active = texture_node
                        for area in screen.areas:
                            if area.type == 'IMAGE_EDITOR' and not area.spaces.active.use_image_pin:
                                area.spaces.active.image = image
                        scene.dream_textures_prompt.seed = str(result.seed) # update property in case seed was sourced randomly or from hash
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
                        history_entry.seed = str(result.seed)
                        history_entry.hash = image_hash
                        history_entry.width = result.image.shape[1]
                        history_entry.height = result.image.shape[0]
                        if is_file_batch:
                            history_entry.prompt_structure_token_subject = file_batch_lines[iteration]
                        iteration += 1
                    if iteration < iteration_limit:
                        generate_next()
                    else:
                        scene.dream_textures_info = ""
                        scene.dream_textures_progress = 0
                        CancelGenerator.should_continue = None
        
            # Call the backend
            CancelGenerator.should_continue = True # reset global cancellation state
            def generate_next():
                args = prompt.generate_args(context, iteration=iteration, init_image=init_image, control_images=control_images)
                backend.generate(args, step_callback=step_callback, callback=callback)
            
            generate_next()
        
        # Prepare ControlNet images
        if len(prompt.control_nets) > 0:
            bpy.types.Scene.dream_textures_progress = bpy.props.IntProperty(name="", default=0, min=0, max=len(prompt.control_nets))
            scene.dream_textures_info = "Processing Control Images..."
            context.scene.dream_textures_progress = 0

            gen = Generator.shared()
            optimizations = backend.optimizations() if isinstance(backend, DiffusersBackend) else Optimizations()

            control_images = []
            def process_next(i):
                if i >= len(prompt.control_nets):
                    execute_backend(control_images)
                    return
                net = prompt.control_nets[i]
                future = gen.controlnet_aux(
                    processor_id=net.processor_id,
                    image=image_utils.bpy_to_np(net.control_image, color_space=None),
                    optimizations=optimizations
                )
                def on_response(future):
                    control_images.append(future.result(last_only=True))
                    context.scene.dream_textures_progress = i + 1
                    process_next(i + 1)
                future.add_done_callback(on_response)
            process_next(0)
        else:
            execute_backend(None)

        return {"FINISHED"}

def kill_generator(context=bpy.context):
    Generator.shared_close()
    try:
        context.scene.dream_textures_info = ""
        context.scene.dream_textures_progress = 0
        CancelGenerator.should_continue = None
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

    should_continue = None

    @classmethod
    def poll(cls, context):
        return cls.should_continue is not None

    def execute(self, context):
        CancelGenerator.should_continue = False
        return {'FINISHED'}
