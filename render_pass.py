import bpy
import cycles
import threading
import functools
import numpy as np
import math
import os
import sys
from multiprocessing.shared_memory import SharedMemory

from .generator_process import GeneratorProcess

from .operators.dream_texture import dream_texture, weights_are_installed

update_render_passes_original = cycles.CyclesRender.update_render_passes
render_original = cycles.CyclesRender.render
del_original = cycles.CyclesRender.__del__

def register_render_pass():
    def update_render_passes_decorator(original):
        def update_render_passes(self, scene=None, renderlayer=None):
            result = original(self, scene, renderlayer)
            self.register_pass(scene, renderlayer, "Dream Textures", 4, "RGBA", 'COLOR')
            self.register_pass(scene, renderlayer, "Dream Textures Seed", 1, "X", 'VALUE')
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
                self.add_pass("Dream Textures Seed", 1, "X")
                scale = scene.render.resolution_percentage / 100.0
                size_x = int(scene.render.resolution_x * scale)
                size_y = int(scene.render.resolution_y * scale)
                if size_x % 64 != 0 or size_y % 64 != 0:
                    self.report({"ERROR"}, f"Image dimensions must be multiples of 64 (e.x. 512x512, 512x768, ...) closest is {round(size_x/64)*64}x{round(size_y/64)*64}")
                    return result
                if not weights_are_installed(self.report):
                    return result
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
                        def do_ocio_transform(event, target_pixels, target_pixels_memory, inverse):
                            ocio_config_path = os.path.join(bpy.utils.resource_path('LOCAL'), 'datafiles/colormanagement/config.ocio')
                            args = {
                                'config_path': ocio_config_path,
                                'name': target_pixels_memory.name,

                                'exposure': scene.view_settings.exposure,
                                'gamma': scene.view_settings.gamma,
                                'view_transform': scene.view_settings.view_transform,
                                'display_device': scene.display_settings.display_device,
                                'look': scene.view_settings.look,
                                
                                'inverse': inverse
                            }
                            def image_callback(event, shared_memory_name, seed, width, height, upscaled=False):
                                nonlocal target_pixels
                                nonlocal target_pixels_memory
                                target_pixels[:] = np.frombuffer(target_pixels_memory.buf, dtype=np.float32).copy().reshape((size_x * size_y, 4))
                                event.set()
                            def exception_callback(fatal, msg, trace):
                                print(fatal, msg, trace)
                                event.set()
                            generator_advance = GeneratorProcess.shared().apply_ocio_transforms(args, functools.partial(image_callback, event), exception_callback)
                            def timer():
                                try:
                                    next(generator_advance)
                                    return 0.01
                                except StopIteration:
                                    pass
                            bpy.app.timers.register(timer)
                        if render_pass.name == "Dream Textures":
                            self.update_stats("Dream Textures", "Starting")
                            def image_callback(event, set_pixels, shared_memory_name, seed, width, height, upscaled=False):
                                self.update_stats("Dream Textures", "Pushing to render pass")
                                # Only use the non-upscaled texture, as upscaling is currently unsupported by the addon.
                                if not upscaled:
                                    shared_memory = SharedMemory(shared_memory_name)
                                    set_pixels(np.frombuffer(shared_memory.buf, dtype=np.float32).copy().reshape((size_x * size_y, 4)))

                                    seed_pass = next(filter(lambda x: x.name == "Dream Textures Seed", layer.passes))
                                    seed_pass_data = np.repeat(np.float32(float(seed)), len(seed_pass.rect))
                                    seed_pass.rect.foreach_set(seed_pass_data)

                                    shared_memory.close()

                                    event.set()
                            
                            step_count = int(scene.dream_textures_render_properties_prompt.strength * scene.dream_textures_render_properties_prompt.steps)
                            def step_callback(step, width=None, height=None, shared_memory_name=None):
                                self.update_stats("Dream Textures", f"Step {step + 1}/{step_count}")
                                self.update_progress(step / step_count)
                                return
                            
                            self.update_stats("Dream Textures", "Creating temporary image")
                            combined_pass_image = bpy.data.images.new("dream_textures_post_processing_temp", width=size_x, height=size_y)
                            
                            rect = layer.passes["Combined"].rect
                            
                            combined_pixels = np.empty((size_x * size_y, 4), dtype=np.float32)
                            rect.foreach_get(combined_pixels)

                            event = threading.Event()

                            buf = combined_pixels.tobytes()
                            combined_pixels_memory = SharedMemory(create=True, size=len(buf))
                            combined_pixels_memory.buf[:] = buf
                            bpy.app.timers.register(functools.partial(do_ocio_transform, event, combined_pixels, combined_pixels_memory, False))
                            event.wait()
                            
                            combined_pass_image.pixels[:] = combined_pixels.ravel()

                            self.update_stats("Dream Textures", "Starting...")
                            event = threading.Event()
                            pixels = None
                            def set_pixels(npbuf):
                                nonlocal pixels
                                pixels = npbuf
                            def do_dream_texture_pass():
                                dream_texture(scene.dream_textures_render_properties_prompt, step_callback, functools.partial(image_callback, event, set_pixels), combined_pass_image, width=size_x, height=size_y, show_steps=False)
                            bpy.app.timers.register(do_dream_texture_pass)
                            event.wait()

                            # Perform an inverse transform so when Blender applies its transform everything looks correct.
                            event = threading.Event()
                            buf = pixels.tobytes()
                            combined_pixels_memory.buf[:] = buf
                            bpy.app.timers.register(functools.partial(do_ocio_transform, event, pixels, combined_pixels_memory, True))
                            event.wait()

                            reshaped = pixels.reshape((size_x * size_y, 4))
                            render_pass.rect.foreach_set(reshaped)

                            # delete pointers before closing shared memory
                            del pixels
                            del combined_pixels
                            del reshaped

                            combined_pixels_memory.close()

                            def cleanup():
                                bpy.data.images.remove(combined_pass_image)
                            bpy.app.timers.register(cleanup)
                            self.update_stats("Dream Textures", "Finished")
                        elif render_pass.name != "Dream Textures Seed":
                            pixels = np.empty((len(original_render_pass.rect), len(original_render_pass.rect[0])), dtype=np.float32)
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

def unregister_render_pass():
    global update_render_passes_original
    cycles.CyclesRender.update_render_passes = update_render_passes_original
    global render_original
    cycles.CyclesRender.render = render_original
    # global del_original
    # cycles.CyclesRender.__del__ = del_original
