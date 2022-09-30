# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name": "Dream Textures",
    "author": "Carson Katri, Greg Richardson, Kevin C. Burke",
    "description": "Use Stable Diffusion to generate unique textures straight from the shader editor.",
    "warning": "Requires installation of Stable Diffusion model weights",
    "blender": (3, 0, 0),
    "version": (0, 0, 6),
    "location": "",
    "warning": "",
    "category": "Paint"
}

from time import time
import bpy
from bpy.props import IntProperty, PointerProperty, EnumProperty, BoolProperty
import sys

import cycles
import numpy as np
import gpu

from .handlers.render_post import render_post_handler

from .help_section import register_section_props

from .prompt_engineering import *
from .operators.open_latest_version import check_for_updates
from .classes import CLASSES, PREFERENCE_CLASSES
from .tools import TOOLS
from .operators.dream_texture import dream_texture, kill_generator, create_generator, get_generator
from .generator_process import GeneratorProcess
from .property_groups.dream_prompt import DreamPrompt

requirements_path_items = (
    # Use the old version of requirements-win.txt to fix installation issues with Blender + PyTorch 1.12.1
    ('requirements-win-torch-1-11-0.txt', 'Linux/Windows (CUDA)', 'Linux or Windows with NVIDIA GPU'),
    ('stable_diffusion/requirements-mac-MPS-CPU.txt', 'Apple Silicon', 'Apple M1/M2'),
    ('stable_diffusion/requirements-lin-AMD.txt', 'Linux (AMD)', 'Linux with AMD GPU'),
)

update_render_passes_original = cycles.CyclesRender.update_render_passes
render_original = cycles.CyclesRender.render

def register():
    bpy.types.Scene.dream_textures_requirements_path = EnumProperty(name="Platform", items=requirements_path_items, description="Specifies which set of dependencies to install", default='stable_diffusion/requirements-mac-MPS-CPU.txt' if sys.platform == 'darwin' else 'requirements-win-torch-1-11-0.txt')
    
    register_section_props()

    for cls in PREFERENCE_CLASSES:
        bpy.utils.register_class(cls)

    check_for_updates()

    bpy.types.Scene.dream_textures_prompt = PointerProperty(type=DreamPrompt)
    bpy.types.Scene.init_img = PointerProperty(name="Init Image", type=bpy.types.Image)
    bpy.types.Scene.init_mask = PointerProperty(name="Init Mask", type=bpy.types.Image)
    bpy.types.Scene.dream_textures_history_selection = IntProperty()
    bpy.types.Scene.dream_textures_progress = bpy.props.IntProperty(name="Progress", default=0, min=0, max=0)
    bpy.types.Scene.dream_textures_info = bpy.props.StringProperty(name="Info")

    bpy.types.Scene.dream_textures_render_properties_enabled = BoolProperty(default=False)
    bpy.types.Scene.dream_textures_render_properties_prompt = PointerProperty(type=DreamPrompt)

    bpy.app.handlers.render_post.append(render_post_handler)

    for cls in CLASSES:
        bpy.utils.register_class(cls)

    for tool in TOOLS:
        bpy.utils.register_tool(tool)

    # Monkey patch cycles render passes
    def update_render_passes_decorator(original):
        def update_render_passes(self, scene=None, renderlayer=None):
            result = original(self, scene, renderlayer)
            self.register_pass(scene, renderlayer, "Dream Textures", 4, "RGBA", 'COLOR')
            return result
        return update_render_passes
    cycles.CyclesRender.update_render_passes = update_render_passes_decorator(cycles.CyclesRender.update_render_passes)
    def render_decorator(original):
        def render(self, depsgraph):
            scene = depsgraph.scene if hasattr(depsgraph, "scene") else depsgraph
            result = original(self, depsgraph)
            try:
                # TODO: Create the render result from scratch
                original_result = self.get_result()
                self.add_pass("Dream Textures", 4, "RGBA")
                scale = scene.render.resolution_percentage / 100.0
                size_x = int(scene.render.resolution_x * scale)
                size_y = int(scene.render.resolution_y * scale)
                render_result = self.begin_result(0, 0, size_x, size_y)
                for original_layer in original_result.layers:
                    layer = None
                    for layer_i in render_result.layers:
                        if layer_i.name == original_layer.name:
                            layer = layer_i
                    for original_render_pass in original_layer.passes:
                        render_pass = None
                        for pass_i in layer.passes:
                            if pass_i.name == original_render_pass.name:
                                render_pass = pass_i
                        if render_pass.name == "Dream Textures":
                            print("Generating dream textures pass")
                            def image_callback(seed, width, height, pixels, upscaled=False):
                                print("Image Callback")
                                # Only use the non-upscaled texture, as upscaling is currently unsupported by the addon.
                                if not upscaled:
                                    reshaped = pixels.reshape((width * height, 4))
                                    render_pass.rect.foreach_set(reshaped)
                                    print("Updated render pass")
                                    get_generator().process.terminate()
                            
                            def step_callback(step, width=None, height=None, pixels=None):
                                print(step)
                                return
                            
                            scale = scene.render.resolution_percentage / 100.0
                            size_x = int(scene.render.resolution_x * scale)
                            size_y = int(scene.render.resolution_y * scale)
                            
                            combined_pass_image = bpy.data.images.new("dream_textures_post_processing_temp", width=size_x, height=size_y)
                            
                            rect = layer.passes["Combined"].rect
                            
                            combined_pixels = np.empty((size_x * size_y, 4), dtype=np.float32)
                            rect.foreach_get(combined_pixels)
                            combined_pass_image.pixels[:] = combined_pixels.ravel()

                            print("Render...")
                            if get_generator() is None:
                                create_generator()
                            def do_dream_texture_pass():
                                dream_texture(scene.dream_textures_render_properties_prompt, step_callback, image_callback, combined_pass_image)
                            bpy.app.timers.register(do_dream_texture_pass)
                            # get_generator().process.communicate()
                            print("waiting for process")
                            exit_code = get_generator().process.wait()
                            print(exit_code)
                            # print("waiting for thread")
                            # get_generator().thread.join()
                            # print("Thread finished")

                            # from .absolute_path import absolute_path
                            # import os
                            # import site
                            # # Support Apple Silicon GPUs as much as possible.
                            # os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                            # sys.path.append(absolute_path("stable_diffusion/"))
                            # sys.path.append(absolute_path("stable_diffusion/src/clip"))
                            # sys.path.append(absolute_path("stable_diffusion/src/k-diffusion"))
                            # sys.path.append(absolute_path("stable_diffusion/src/taming-transformers"))

                            # site.addsitedir(absolute_path(".python_dependencies"))
                            # print("Added package dirs")
                            # import pkg_resources
                            # pkg_resources._initialize_master_working_set()
                            # print("Reloaded working set")

                            # from stable_diffusion.ldm.generate import Generate
                            # print("Import generate")
                            # from omegaconf import OmegaConf
                            # print("Import omegaconf")
                            # from PIL import ImageOps
                            # print("Import PIL.ImageOps")

                            # models_config  = absolute_path('stable_diffusion/configs/models.yaml')
                            # model   = 'stable-diffusion-1.4'

                            # models  = OmegaConf.load(models_config)
                            # config  = absolute_path('stable_diffusion/' + models[model].config)
                            # weights = absolute_path('stable_diffusion/' + models[model].weights)

                            # byte_to_normalized = 1.0 / 255.0
                            # def write_pixels(image):
                            #     pixels = (np.asarray(ImageOps.flip(image).convert('RGBA'),dtype=np.float32) * byte_to_normalized).reshape((image.width * image.height, 4))
                            #     render_pass.rect.foreach_set(pixels)
                            # def image_writer(image, seed, upscaled=False):
                            #     # Only use the non-upscaled texture, as upscaling is currently unsupported by the addon.
                            #     if not upscaled:
                            #         write_pixels(image)
                            
                            # def view_step(samples, step):
                            #     print(step)

                            # def save_init_img():
                            #     import tempfile
                            #     from PIL import Image
                            #     path = path if path is not None else tempfile.NamedTemporaryFile().name
                                
                            #     image = Image.fromarray(combined_pixels.ravel())
                            #     image.save(path, 'PNG')
                            #     return path
                            
                            # print("Loading Model")
                            # generator = Generate(
                            #     conf=models_config,
                            #     model=model,
                            #     # These args are deprecated, but we need them to specify an absolute path to the weights.
                            #     weights=weights,
                            #     config=config,
                            #     full_precision=args['full_precision']
                            # )
                            # generator.load_model()

                            # args = {key: getattr(scene.dream_textures_render_properties_prompt,key) for key in DreamPrompt.__annotations__}
                            # args['prompt'] = scene.dream_textures_render_properties_prompt.generate_prompt()
                            # args['seed'] = scene.dream_textures_render_properties_prompt.get_seed()
                            # args['use_init_img'] = True
                            # args['init_img'] = save_init_img()
                            # args['show_steps'] = False

                            # generator.prompt2image(
                            #     # a function or method that will be called each step
                            #     step_callback=view_step,
                            #     # a function or method that will be called each time an image is generated
                            #     image_callback=image_writer,
                            #     **args
                            # )
                        else:
                            render_pass.rect[:] = original_render_pass.rect
                self.end_result(render_result)
            except Exception as e:
                print(e)
            return result
        return render
    cycles.CyclesRender.render = render_decorator(cycles.CyclesRender.render)

def unregister():
    try:
        bpy.app.handlers.render_post.remove(render_post_handler)
    except:
        pass
    
    for cls in PREFERENCE_CLASSES:
        bpy.utils.unregister_class(cls)

    for cls in CLASSES:
        bpy.utils.unregister_class(cls)
    for tool in TOOLS:
        bpy.utils.unregister_tool(tool)
    
    global update_render_passes_original
    cycles.CyclesRender.update_render_passes = update_render_passes_original
    global render_original
    cycles.CyclesRender.render = render_original

    kill_generator()

if __name__ == "__main__":
    register()