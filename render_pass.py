import bpy
import cycles
import numpy as np
import os
from .generator_process.actions.prompt_to_image import Pipeline, StepPreviewMode
from .generator_process import Generator

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
                            
                            self.update_stats("Dream Textures", "Creating temporary image")
                            
                            rect = layer.passes["Combined"].rect
                            
                            combined_pixels = np.empty((size_x * size_y, 4), dtype=np.float32)
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
                                inverse=False,
                                _block=True
                            ).result()

                            self.update_stats("Dream Textures", "Generating...")
                            
                            generated_args = scene.dream_textures_render_properties_prompt.generate_args()
                            generated_args['step_preview_mode'] = None
                            generated_args['width'] = size_x
                            generated_args['height'] = size_y
                            combined_pixels = gen.image_to_image(
                                image=np.flipud(combined_pixels.reshape((size_y, size_x, 4)) * 255).astype(np.uint8),
                                **generated_args,
                                _block=True
                            ).result().image

                            # Perform an inverse transform so when Blender applies its transform everything looks correct.
                            self.update_stats("Dream Textures", "Applying inverse color management transforms")
                            combined_pixels = gen.ocio_transform(
                                combined_pixels.reshape((size_x * size_y, 4)),
                                config_path=os.path.join(bpy.utils.resource_path('LOCAL'), 'datafiles/colormanagement/config.ocio'),
                                exposure=scene.view_settings.exposure,
                                gamma=scene.view_settings.gamma,
                                view_transform=scene.view_settings.view_transform,
                                display_device=scene.display_settings.display_device,
                                look=scene.view_settings.look,
                                inverse=True,
                                _block=True
                            ).result()

                            combined_pixels = combined_pixels.reshape((size_x * size_y, 4))
                            render_pass.rect.foreach_set(combined_pixels)

                            self.update_stats("Dream Textures", "Finished")
                        else:
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
