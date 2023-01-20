import bpy
import cycles
import numpy as np
import os
from .generator_process.actions.prompt_to_image import ImageGenerationResult
from .generator_process import Generator
import threading

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
                            pixels = np.empty((len(source_pass.rect), len(source_pass.rect[0])), dtype=np.float32)
                            source_pass.rect.foreach_get(pixels)
                            render_pass.rect[:] = pixels
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
    self.update_stats("Dream Textures", "Starting")
    
    rect = layer.passes["Combined"].rect

    match scene.dream_textures_render_properties_pass_inputs:
        case 'color': pass
        case 'depth' | 'color_depth':
            depth = np.empty((size[0] * size[1], 1), dtype=np.float32)
            layer.passes["Depth"].rect.foreach_get(depth)
            depth = (1 - np.interp(depth, [0, np.ma.masked_equal(depth, depth.max(), copy=False).max()], [0, 1])).reshape((size[1], size[0]))
    
    combined_pixels = np.empty((size[0] * size[1], 4), dtype=np.float32)
    rect.foreach_get(combined_pixels)

    gen = Generator.shared()
    self.update_stats("Dream Textures", "Applying color management transforms")
    combined_pixels = gen.ocio_transform(
        combined_pixels,
        config_path=os.path.join(bpy.utils.resource_path('LOCAL'), 'datafiles/colormanagement/config.ocio'),
        exposure=scene.view_settings.exposure,
        gamma=scene.view_settings.gamma,
        view_transform=scene.view_settings.view_transform,
        display_device=scene.display_settings.display_device,
        look=scene.view_settings.look,
        inverse=False
    ).result()

    self.update_stats("Dream Textures", "Generating...")
    
    generated_args = scene.dream_textures_render_properties_prompt.generate_args()
    generated_args['width'] = size[0]
    generated_args['height'] = size[1]
    match scene.dream_textures_render_properties_pass_inputs:
        case 'color':
            f = gen.image_to_image(
                image=np.flipud(combined_pixels.reshape((size[1], size[0], 4)) * 255).astype(np.uint8),
                **generated_args
            )
        case 'depth':
            f = gen.depth_to_image(
                depth=depth,
                image=None,
                **generated_args
            )
        case 'color_depth':
            f = gen.depth_to_image(
                depth=depth,
                image=np.flipud(combined_pixels.reshape((size[1], size[0], 4)) * 255).astype(np.uint8),
                **generated_args
            )
    event = threading.Event()
    def on_step(_, step: ImageGenerationResult):
        if step.final:
            return
        self.update_progress(step.step / generated_args['steps'])
        if len(step.images) > 0:
            combined_pixels = step.images[0]
            render_pass.rect.foreach_set(combined_pixels.reshape((size[0] * size[1], 4)))
            self.update_result(render_result) # This does not seem to have an effect.
    def on_done(future):
        nonlocal combined_pixels
        result = future.result(last_only=True)
        combined_pixels = result.images[0]
        event.set()
    f.add_response_callback(on_step)
    f.add_done_callback(on_done)
    event.wait()

    # Perform an inverse transform so when Blender applies its transform everything looks correct.
    self.update_stats("Dream Textures", "Applying inverse color management transforms")
    combined_pixels = gen.ocio_transform(
        combined_pixels.reshape((size[0] * size[1], 4)),
        config_path=os.path.join(bpy.utils.resource_path('LOCAL'), 'datafiles/colormanagement/config.ocio'),
        exposure=scene.view_settings.exposure,
        gamma=scene.view_settings.gamma,
        view_transform=scene.view_settings.view_transform,
        display_device=scene.display_settings.display_device,
        look=scene.view_settings.look,
        inverse=True
    ).result()

    combined_pixels = combined_pixels.reshape((size[0] * size[1], 4))
    render_pass.rect.foreach_set(combined_pixels)

    self.update_stats("Dream Textures", "Finished")