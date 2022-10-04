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

import bpy
from bpy.props import IntProperty, PointerProperty, EnumProperty, BoolProperty
import sys

import cycles
import threading
import functools
import numpy as np

from .prompt_engineering import *
from .operators.open_latest_version import check_for_updates
from .classes import CLASSES, PREFERENCE_CLASSES
from .tools import TOOLS
from .operators.dream_texture import DreamTexture, kill_generator, dream_texture
from .property_groups.dream_prompt import DreamPrompt
from .operators.upscale import upscale_options
from .preferences import StableDiffusionPreferences

requirements_path_items = (
    # Use the old version of requirements-win.txt to fix installation issues with Blender + PyTorch 1.12.1
    ('requirements-win-torch-1-11-0.txt', 'Linux/Windows (CUDA)', 'Linux or Windows with NVIDIA GPU'),
    ('stable_diffusion/requirements-mac-MPS-CPU.txt', 'Apple Silicon', 'Apple M1/M2'),
    ('stable_diffusion/requirements-lin-AMD.txt', 'Linux (AMD)', 'Linux with AMD GPU'),
)

update_render_passes_original = cycles.CyclesRender.update_render_passes
render_original = cycles.CyclesRender.render
del_original = cycles.CyclesRender.__del__

def register():
    dt_op = bpy.ops
    for name in DreamTexture.bl_idname.split("."):
        dt_op = getattr(dt_op, name)
    if hasattr(bpy.types, dt_op.idname()): # objects under bpy.ops are created on the fly, have to check that it actually exists a little differently
        raise RuntimeError("Another instance of Dream Textures is already running.")
    

    bpy.types.Scene.dream_textures_requirements_path = EnumProperty(name="Platform", items=requirements_path_items, description="Specifies which set of dependencies to install", default='stable_diffusion/requirements-mac-MPS-CPU.txt' if sys.platform == 'darwin' else 'requirements-win-torch-1-11-0.txt')

    for cls in PREFERENCE_CLASSES:
        bpy.utils.register_class(cls)

    check_for_updates()

    bpy.types.Scene.dream_textures_prompt = PointerProperty(type=DreamPrompt)
    bpy.types.Scene.init_img = PointerProperty(name="Init Image", type=bpy.types.Image)
    bpy.types.Scene.init_mask = PointerProperty(name="Init Mask", type=bpy.types.Image)
    def update_selection_preview(self, context):
        history = context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences.history
        if context.scene.dream_textures_history_selection > 0 and context.scene.dream_textures_history_selection < len(history):
            context.scene.dream_textures_history_selection_preview = history[context.scene.dream_textures_history_selection].generate_prompt()
    bpy.types.Scene.dream_textures_history_selection = IntProperty(update=update_selection_preview)
    bpy.types.Scene.dream_textures_history_selection_preview = bpy.props.StringProperty(name="", default="", update=update_selection_preview)
    bpy.types.Scene.dream_textures_progress = bpy.props.IntProperty(name="Progress", default=0, min=0, max=0)
    bpy.types.Scene.dream_textures_info = bpy.props.StringProperty(name="Info")

    bpy.types.Scene.dream_textures_render_properties_enabled = BoolProperty(default=False)
    bpy.types.Scene.dream_textures_render_properties_prompt = PointerProperty(type=DreamPrompt)
    bpy.types.Scene.dream_textures_upscale_outscale = bpy.props.EnumProperty(name="Target Size", items=upscale_options)
    bpy.types.Scene.dream_textures_upscale_full_precision = bpy.props.BoolProperty(name="Full Precision", default=True)

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
            if not scene.dream_textures_render_properties_enabled:
                return original(self, depsgraph)
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
                            self.update_stats("Dream Textures", "Starting")
                            def image_callback(event, seed, width, height, pixels, upscaled=False):
                                self.update_stats("Dream Textures", "Pushing to render pass")
                                # Only use the non-upscaled texture, as upscaling is currently unsupported by the addon.
                                if not upscaled:
                                    reshaped = pixels.reshape((width * height, 4))
                                    render_pass.rect.foreach_set(reshaped)
                                    event.set()
                                    # kill_generator() # Cleanup to avoid memory leak
                            
                            step_count = scene.dream_textures_render_properties_prompt.steps
                            def step_callback(step, width=None, height=None, pixels=None):
                                self.update_stats("Dream Textures", f"Step {step}/{step_count}")
                                self.update_progress(step / step_count)
                                return
                            
                            self.update_stats("Dream Textures", "Creating temporary image")
                            combined_pass_image = bpy.data.images.new("dream_textures_post_processing_temp", width=size_x, height=size_y)
                            
                            rect = layer.passes["Combined"].rect
                            
                            combined_pixels = np.empty((size_x * size_y, 4), dtype=np.float32)
                            rect.foreach_get(combined_pixels)
                            combined_pass_image.pixels[:] = combined_pixels.ravel()

                            self.update_stats("Dream Textures", "Starting...")
                            event = threading.Event()
                            def do_dream_texture_pass():
                                dream_texture(scene.dream_textures_render_properties_prompt, step_callback, functools.partial(image_callback, event), combined_pass_image)
                            bpy.app.timers.register(do_dream_texture_pass)
                            event.wait()
                            # bpy.data.images.remove(combined_pass_image)
                            self.update_stats("Dream Textures", "Finished")
                        else:
                            pixels = np.empty((size_x * size_y, 4), dtype=np.float32)
                            original_render_pass.rect.foreach_get(pixels)
                            render_pass.rect[:] = pixels
                self.end_result(render_result)
            except Exception as e:
                print(e)
            return result
        return render
    cycles.CyclesRender.render = render_decorator(cycles.CyclesRender.render)
    # def del_decorator(original):
    #     def del_patch(self):
    #         result = original(self)
    #         kill_generator()
    #         return result
    #     return del_patch
    # cycles.CyclesRender.__del__ = del_decorator(cycles.CyclesRender.__del__)

def unregister():
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
    # global del_original
    # cycles.CyclesRender.__del__ = del_original

    kill_generator()

if __name__ == "__main__":
    register()