import bpy
import cycles
import threading
import functools
import numpy as np
import math
import os
import sys
import PyOpenColorIO as OCIO
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
                            def image_callback(event, shared_memory_name, seed, width, height, upscaled=False):
                                self.update_stats("Dream Textures", "Pushing to render pass")
                                # Only use the non-upscaled texture, as upscaling is currently unsupported by the addon.
                                if not upscaled:
                                    shared_memory = SharedMemory(shared_memory_name)
                                    pixels = np.frombuffer(shared_memory.buf, dtype=np.float32)
                                    reshaped = np.power(pixels.reshape((width * height, 4)), 2.4)
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

                            ocio_config = OCIO.Config.CreateFromFile(os.path.join(bpy.utils.resource_path('LOCAL'), 'datafiles/colormanagement/config.ocio'))
                            
                            # A reimplementation of `OCIOImpl::createDisplayProcessor` from the Blender source.
                            # https://github.com/dfelinto/blender/blob/87a0770bb969ce37d9a41a04c1658ea09c63933a/intern/opencolorio/ocio_impl.cc#L643
                            def create_display_processor(
                                config,
                                input_colorspace,
                                view,
                                display,
                                look,
                                scale, # Exposure
                                exponent # Gamma
                            ):
                                group = OCIO.GroupTransform()

                                # Exposure
                                if scale != 1:
                                    # Always apply exposure in scene linear.
                                    color_space_transform = OCIO.ColorSpaceTransform()
                                    color_space_transform.setSrc(input_colorspace)
                                    color_space_transform.setDst(OCIO.ROLE_SCENE_LINEAR)
                                    group.appendTransform(color_space_transform)

                                    # Make further transforms aware of the color space change
                                    input_colorspace = OCIO.ROLE_SCENE_LINEAR

                                    # Apply scale
                                    matrix_transform = OCIO.MatrixTransform([scale, 0.0, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 0.0, 1.0])
                                    group.appendTransform(matrix_transform)
                                
                                # Add look transform
                                use_look = look is not None and len(look) > 0
                                if use_look:
                                    look_output = config.getLook(look).getProcessSpace()
                                    if look_output is not None and len(look_output) > 0:
                                        look_transform = OCIO.LookTransform()
                                        look_transform.setSrc(input_colorspace)
                                        look_transform.setDst(look_output)
                                        look_transform.setLooks(look)
                                        group.appendTransform(look_transform)
                                        # Make further transforms aware of the color space change.
                                        input_colorspace = look_output
                                    else:
                                        # For empty looks, no output color space is returned.
                                        use_look = False
                                
                                # Add view and display transform
                                print(input_colorspace)
                                print(use_look)
                                print(view)
                                print(display)
                                display_view_transform = OCIO.DisplayViewTransform()
                                display_view_transform.setSrc(input_colorspace)
                                display_view_transform.setLooksBypass(True)
                                display_view_transform.setView(view)
                                display_view_transform.setDisplay(display)
                                group.appendTransform(display_view_transform)

                                # Gamma
                                if exponent != 1:
                                    exponent_transform = OCIO.ExponentTransform([exponent, exponent, exponent, 1.0])
                                    group.appendTransform(exponent_transform)
                                
                                # Create processor from transform. This is the moment were OCIO validates
                                # the entire transform, no need to check for the validity of inputs above.
                                try:
                                    processor = config.getProcessor(group)
                                    if processor is not None:
                                        return processor
                                except Exception as e:
                                    print(e)
                                
                                return None
                            
                            # Exposure and gamma transformations derived from Blender source:
                            # https://github.com/dfelinto/blender/blob/87a0770bb969ce37d9a41a04c1658ea09c63933a/source/blender/imbuf/intern/colormanagement.c#L825
                            scale = 1 if scene.view_settings.exposure == 0 else math.pow(2, scene.view_settings.exposure)
                            exponent = 1 if scene.view_settings.gamma == 1 else (1 / (scene.view_settings.gamma if scene.view_settings.gamma > sys.float_info.epsilon else sys.float_info.epsilon))
                            processor = create_display_processor(ocio_config, OCIO.ROLE_SCENE_LINEAR, scene.view_settings.view_transform, scene.display_settings.display_device, scene.view_settings.look if scene.view_settings.look != 'None' else None, scale, exponent)

                            flat_pixels = combined_pixels.ravel()
                            processor.getDefaultCPUProcessor().applyRGBA(flat_pixels)
                            combined_pass_image.pixels[:] = flat_pixels

                            self.update_stats("Dream Textures", "Starting...")
                            event = threading.Event()
                            def do_dream_texture_pass():
                                dream_texture(scene.dream_textures_render_properties_prompt, step_callback, functools.partial(image_callback, event), combined_pass_image, width=size_x, height=size_y, show_steps=False)
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