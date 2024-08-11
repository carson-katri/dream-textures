import bpy
import cycles
import numpy as np
import os
from typing import List
import threading
from .generator_process import Generator
from . import api
from . import image_utils

pass_inputs = [
    ('color', 'Color', 'Provide the scene color as input'),
    ('depth', 'Depth', 'Provide the Z pass as depth input'),
    ('color_depth', 'Color and Depth', 'Provide the scene color and depth as input'),
]

update_render_passes_original = cycles.CyclesRender.update_render_passes
render_original = cycles.CyclesRender.render
# del_original = cycles.CyclesRender.__del__

def register_render_pass():
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
                original_result = self.get_result()
                self.add_pass("Dream Textures", 4, "RGBA")
                scale = scene.render.resolution_percentage / 100.0
                size_x = int(scene.render.resolution_x * scale)
                size_y = int(scene.render.resolution_y * scale)
                if size_x % 64 != 0 or size_y % 64 != 0:
                    self.report({"ERROR"}, f"Image dimensions must be multiples of 64 (e.x. 512x512, 512x768, ...) closest is {round(size_x/64)*64}x{round(size_y/64)*64}")
                    return result
                render_result = self.begin_result(0, 0, size_x, size_y)
                for layer in render_result.layers:
                    for render_pass in layer.passes:
                        if render_pass.name == "Dream Textures":
                            try:
                                self._render_dream_textures_pass(layer, (size_x, size_y), scene, render_pass, render_result)
                            except Exception as e:
                                self.error_set(str(e))
                        else:
                            source_pass = None
                            for original_layer in original_result.layers:
                                if layer.name == original_layer.name:
                                    for original_pass in original_layer.passes:
                                        if original_pass.name == render_pass.name:
                                            source_pass = original_pass
                            pixels = image_utils.render_pass_to_np(source_pass, size=(size_x, size_y))
                            image_utils.np_to_render_pass(pixels, render_pass)
                self.end_result(render_result)
            except Exception as e:
                print(e)
            return result
        return render
    cycles.CyclesRender.render = render_decorator(cycles.CyclesRender.render)
    cycles.CyclesRender._render_dream_textures_pass = _render_dream_textures_pass

    # def del_decorator(original):
    #     def del_patch(self):
    #         result = original(self)
    #         kill_generator()
    #         return result
    #     return del_patch
    # cycles.CyclesRender.__del__ = del_decorator(cycles.CyclesRender.__del__)

def unregister_render_pass():
    global update_render_passes_original
    cycles.CyclesRender.update_render_passes = update_render_passes_original
    global render_original
    cycles.CyclesRender.render = render_original
    del cycles.CyclesRender._render_dream_textures_pass
    # global del_original
    # cycles.CyclesRender.__del__ = del_original

def _render_dream_textures_pass(self, layer, size, scene, render_pass, render_result):
    def combined():
        self.update_stats("Dream Textures", "Applying color management transforms")
        return image_utils.render_pass_to_np(layer.passes["Combined"], size, color_management=True, color_space="sRGB")

    def depth():
        d = image_utils.render_pass_to_np(layer.passes["Depth"], size).squeeze(2)
        return (1 - np.interp(d, [0, np.ma.masked_equal(d, d.max(), copy=False).max()], [0, 1]))

    self.update_stats("Dream Textures", "Starting")
    
    prompt = scene.dream_textures_render_properties_prompt
    match scene.dream_textures_render_properties_pass_inputs:
        case 'color':
            task = api.ImageToImage(
                combined(),
                prompt.strength,
                True
            )
        case 'depth':
            task = api.DepthToImage(
                depth(),
                None,
                prompt.strength
            )
        case 'color_depth':
            task = api.DepthToImage(
                depth(),
                combined(),
                prompt.strength
            )
    event = threading.Event()
    dream_pixels = None
    def step_callback(progress: List[api.GenerationResult]) -> bool:
        self.update_progress(progress[-1].progress / progress[-1].total)
        image_utils.np_to_render_pass(progress[-1].image, render_pass)
        self.update_result(render_result) # This does not seem to have an effect.
        return True
    def callback(results: List[api.GenerationResult] | Exception):
        nonlocal dream_pixels
        dream_pixels = results[-1].image
        event.set()
    
    backend: api.Backend = prompt.get_backend()
    generated_args: api.GenerationArguments = prompt.generate_args(bpy.context)
    generated_args.task = task
    generated_args.size = size
    self.update_stats("Dream Textures", "Generating...")
    backend.generate(
        generated_args,
        step_callback=step_callback,
        callback=callback
    )

    event.wait()

    # Perform an inverse transform so when Blender applies its transform everything looks correct.
    self.update_stats("Dream Textures", "Applying inverse color management transforms")
    image_utils.np_to_render_pass(dream_pixels, render_pass, inverse_color_management=True, color_space="sRGB")

    self.update_stats("Dream Textures", "Finished")