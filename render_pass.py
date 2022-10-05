import bpy
import cycles
import threading
import functools
import numpy as np
from multiprocessing.shared_memory import SharedMemory

from .operators.dream_texture import dream_texture

update_render_passes_original = cycles.CyclesRender.update_render_passes
render_original = cycles.CyclesRender.render
del_original = cycles.CyclesRender.__del__

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
                            def image_callback(event, shared_memory_name, seed, width, height, upscaled=False):
                                self.update_stats("Dream Textures", "Pushing to render pass")
                                # Only use the non-upscaled texture, as upscaling is currently unsupported by the addon.
                                if not upscaled:
                                    shared_memory = SharedMemory(shared_memory_name)
                                    pixels = np.frombuffer(shared_memory.buf, dtype=np.float32)
                                    reshaped = pixels.reshape((width * height, 4))
                                    render_pass.rect.foreach_set(reshaped)
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
                            combined_pass_image.pixels[:] = combined_pixels.ravel()

                            self.update_stats("Dream Textures", "Starting...")
                            event = threading.Event()
                            def do_dream_texture_pass():
                                dream_texture(scene.dream_textures_render_properties_prompt, step_callback, functools.partial(image_callback, event), combined_pass_image)
                            bpy.app.timers.register(do_dream_texture_pass)
                            event.wait()
                            def cleanup():
                                bpy.data.images.remove(combined_pass_image)
                            bpy.app.timers.register(cleanup)
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

def unregister_render_pass():
    global update_render_passes_original
    cycles.CyclesRender.update_render_passes = update_render_passes_original
    global render_original
    cycles.CyclesRender.render = render_original
    # global del_original
    # cycles.CyclesRender.__del__ = del_original