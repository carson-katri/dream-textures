import sys
import bpy
import os
import hashlib
import numpy as np
from numpy.typing import NDArray
from multiprocessing.shared_memory import SharedMemory

from ..property_groups.dream_prompt import backend_options

from ..generator_process.registrar import BackendTarget

from ..preferences import StableDiffusionPreferences
from ..pil_to_image import *
from ..prompt_engineering import *
from ..absolute_path import WEIGHTS_PATH, CLIPSEG_WEIGHTS_PATH
from ..generator_process import Generator
from ..generator_process.actions.prompt_to_image import Pipeline, Optimizations, ImageGenerationResult

import tempfile

generator_advance = None
last_data_block = None
timer = None

def save_temp_image(img, path=None):
    path = path if path is not None else tempfile.NamedTemporaryFile().name

    settings = bpy.context.scene.render.image_settings
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

def bpy_image(name, width, height, pixels):
    image = bpy.data.images.new(name, width=width, height=height)
    image.pixels[:] = pixels
    image.pack()
    return image

class DreamTexture(bpy.types.Operator):
    bl_idname = "shade.dream_texture"
    bl_label = "Dream Texture"
    bl_description = "Generate a texture with AI"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return Generator.shared().can_use()

    def execute(self, context):
        history_entries = []
        is_file_batch = context.scene.dream_textures_prompt.prompt_structure == file_batch_structure.id
        file_batch_lines = []
        if is_file_batch:
            context.scene.dream_textures_prompt.iterations = 1
            file_batch_lines = [line for line in context.scene.dream_textures_prompt_file.lines if len(line.body.strip()) > 0]
            history_entries = [context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences.history.add() for _ in file_batch_lines]
        else:
            history_entries = [context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences.history.add() for _ in range(context.scene.dream_textures_prompt.iterations)]
        for i, history_entry in enumerate(history_entries):
            for prop in context.scene.dream_textures_prompt.__annotations__.keys():
                try:
                    if hasattr(history_entry, prop):
                        setattr(history_entry, prop, getattr(context.scene.dream_textures_prompt, prop))
                except:
                    continue
            if is_file_batch:
                history_entry.prompt_structure = custom_structure.id
                history_entry.prompt_structure_token_subject = file_batch_lines[i].body

        node_tree = context.material.node_tree if hasattr(context, 'material') and hasattr(context.material, 'node_tree') else None
        screen = context.screen
        scene = context.scene

        generated_args = scene.dream_textures_prompt.generate_args()

        init_image = None
        if generated_args['use_init_img']:
            match generated_args['init_img_src']:
                case 'file':
                    init_image = save_temp_image(scene.init_img)
                case 'open_editor':
                    for area in screen.areas:
                        if area.type == 'IMAGE_EDITOR':
                            if area.spaces.active.image is not None:
                                init_image = save_temp_image(area.spaces.active.image)

        last_data_block = None
        def step_callback(_, step_image: ImageGenerationResult):
            nonlocal last_data_block
            if last_data_block is not None:
                bpy.data.images.remove(last_data_block)
            last_data_block = bpy_image(f"Step {step_image.step}/{generated_args['steps']}", step_image.image.shape[1], step_image.image.shape[0], step_image.image.ravel())
            for area in screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    area.spaces.active.image = last_data_block

        iteration = 0
        def done_callback(future):
            nonlocal last_data_block
            nonlocal iteration
            del gen._active_generation_future
            image_result: ImageGenerationResult | list = future.result()
            if isinstance(image_result, list):
                image_result = image_result[-1]
            if last_data_block is not None:
                bpy.data.images.remove(last_data_block)
            image = bpy_image(str(image_result.seed), image_result.image.shape[1], image_result.image.shape[0], image_result.image.ravel())
            if node_tree is not None:
                nodes = node_tree.nodes
                texture_node = nodes.new("ShaderNodeTexImage")
                texture_node.image = image
                nodes.active = texture_node
            for area in screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    area.spaces.active.image = image
            scene.dream_textures_prompt.seed = str(image_result.seed) # update property in case seed was sourced randomly or from hash
            # create a hash from the Blender image datablock to use as unique ID of said image and store it in the prompt history
            # and as custom property of the image. Needs to be a string because the int from the hash function is too large
            image_hash = hashlib.sha256((np.array(image.pixels) * 255).tobytes()).hexdigest()
            image['dream_textures_hash'] = image_hash
            scene.dream_textures_prompt.hash = image_hash
            history_entries[iteration].seed = str(image_result.seed)
            history_entries[iteration].random_seed = False
            history_entries[iteration].hash = image_hash
            iteration += 1
            if iteration < generated_args['iterations']:
                generate_next()

        gen = Generator.shared()
        def generate_next():
            if init_image is not None:
                match generated_args['init_img_action']:
                    case 'modify':
                        f = gen.image_to_image(
                            Pipeline.STABLE_DIFFUSION,
                            image=init_image,
                            **generated_args
                        )
                    case 'inpaint':
                        f = gen.inpaint(
                            Pipeline.STABLE_DIFFUSION,
                            image=init_image,
                            **generated_args
                        )
                    case 'outpaint':
                        raise NotImplementedError()
            else:
                f = gen.prompt_to_image(
                    Pipeline.STABLE_DIFFUSION,
                    **generated_args,
                )
            gen._active_generation_future = f
            f.add_response_callback(step_callback)
            f.add_done_callback(done_callback)
        generate_next()
        return {"FINISHED"}

headless_prompt = None
headless_step_callback = None
headless_image_callback = None
headless_init_img = None
headless_args = None
def dream_texture(prompt, step_callback, image_callback, init_img=None, **kwargs):
    global headless_prompt
    headless_prompt = prompt
    global headless_step_callback
    headless_step_callback = step_callback
    global headless_image_callback
    headless_image_callback = image_callback
    global headless_init_img
    headless_init_img = init_img
    global headless_args
    headless_args = kwargs
    bpy.ops.shade.dream_texture_headless()

class HeadlessDreamTexture(bpy.types.Operator):
    bl_idname = "shade.dream_texture_headless"
    bl_label = "Headless Dream Texture"
    bl_description = "Generate a texture with AI"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return GeneratorProcess.can_use()

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        try:
            next(generator_advance)
        except StopIteration:
            modal_stopped(context)
            return {'FINISHED'}
        except Exception as e:
            modal_stopped(context)
            raise e
        return {'RUNNING_MODAL'}

    def execute(self, context):
        global headless_prompt
        screen = context.screen
        scene = context.scene

        global headless_init_img
        init_img = headless_init_img or (scene.init_img if headless_prompt.use_init_img and headless_prompt.init_img_src == 'file' else None)

        def info(msg=""):
            scene.dream_textures_info = msg
        
        def handle_exception(fatal, msg, trace):
            info() # clear variable
            if fatal:
                kill_generator()
            self.report({'ERROR'},msg)
            if trace:
                print(trace, file=sys.stderr)
            if msg == MISSING_DEPENDENCIES_ERROR:
                from .open_latest_version import do_force_show_download
                do_force_show_download()

        def step_progress_update(self, context):
            if hasattr(context.area, "regions"):
                for region in context.area.regions:
                    if region.type == "UI":
                        region.tag_redraw()
            return None

        bpy.types.Scene.dream_textures_progress = bpy.props.IntProperty(
            name="",
            default=0,
            min=0,
            max=(int(headless_prompt.strength * headless_prompt.steps) if init_img is not None else headless_prompt.steps) + 1,
            update=step_progress_update
        )
        bpy.types.Scene.dream_textures_info = bpy.props.StringProperty(name="Info", update=step_progress_update)
        
        info("Waiting For Process")
        if len(backend_options(self, context)) <= 1:
            headless_prompt.backend = backend_options(self, context)[0][0]
        generator = GeneratorProcess.shared(backend=BackendTarget[headless_prompt.backend])

        if not generator.backend.color_correction():
            headless_prompt.use_init_img_color = False

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

        if headless_prompt.use_init_img and headless_prompt.init_img_src == 'open_editor':
            for area in screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    if area.spaces.active.image is not None:
                        init_img = area.spaces.active.image
        init_img_path = None
        if init_img is not None:
            init_img_path = save_temp_image(init_img)

        args = headless_prompt.generate_args()
        args.update(headless_args)
        if headless_init_img is not None:
            args['use_init_img'] = True
        if args['prompt_structure'] == file_batch_structure.id:
            args['prompt'] = [line.body for line in scene.dream_textures_prompt_file.lines if len(line.body.strip()) > 0]
        args['init_img'] = init_img_path
        if args['use_init_img_color']:
            args['init_color'] = init_img_path
        if args['backend'] == BackendTarget.STABILITY_SDK.name:
            args['dream_studio_key'] = context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences.dream_studio_key

        def step_callback(step, width=None, height=None, shared_memory_name=None):
            global headless_step_callback
            info() # clear variable
            scene.dream_textures_progress = step + 1
            headless_step_callback(step, width, height, shared_memory_name)

        received_noncolorized = False
        def image_callback(shared_memory_name, seed, width, height, upscaled=False):
            global headless_image_callback
            info() # clear variable
            nonlocal received_noncolorized
            if args['use_init_img'] and args['use_init_img_color'] and not received_noncolorized:
                received_noncolorized = True
                return
            received_noncolorized = False
            headless_image_callback(shared_memory_name, seed, width, height, upscaled)

        global generator_advance
        generator_advance = generator.prompt2image(args,
            # a function or method that will be called each step
            step_callback=step_callback,
            # a function or method that will be called each time an image is generated
            image_callback=image_callback,
            # a function or method that will recieve messages
            info_callback=info,
            exception_callback=handle_exception
        )
        context.window_manager.modal_handler_add(self)
        global timer
        timer = context.window_manager.event_timer_add(1 / 15, window=context.window)
        return {'RUNNING_MODAL'}

def modal_stopped(context):
    global timer
    if timer:
        context.window_manager.event_timer_remove(timer)
        timer = None
    if not hasattr(context,'scene'):
        context = bpy.context # modal context is sometimes missing scene?
    context.scene.dream_textures_progress = 0
    context.scene.dream_textures_info = ""
    global last_data_block
    if last_data_block is not None:
        bpy.data.images.remove(last_data_block)
        last_data_block = None

def kill_generator(context=bpy.context):
    Generator.shared_close()
    modal_stopped(context)

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
    def poll(self, context):
        gen = Generator.shared()
        return hasattr(gen, "_active_generation_future") and gen._active_generation_future is not None and not gen._active_generation_future.cancelled and not gen._active_generation_future.done

    def execute(self, context):
        gen = Generator.shared()
        gen._active_generation_future.cancel()
        return {'FINISHED'}
