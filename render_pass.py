import bpy
import cycles
import gpu
from gpu_extras.batch import batch_for_shader
import threading
import threading
import functools
import numpy as np
import os
import time
from multiprocessing.shared_memory import SharedMemory

from .generator_process import GeneratorProcess

from .operators.dream_texture import dream_texture, weights_are_installed

update_render_passes_original = cycles.CyclesRender.update_render_passes
render_original = cycles.CyclesRender.render
del_original = cycles.CyclesRender.__del__
view_update_original = cycles.CyclesRender.view_update
view_draw_original = cycles.CyclesRender.view_draw

def debounce(wait_time):
    """
    Decorator that will debounce a function so that it is called after wait_time seconds
    If it is called multiple times, will wait for the last call to be debounced and run only this one.
    """

    def decorator(function):
        def debounced(*args, **kwargs):
            def call_function():
                debounced._timer = None
                return function(*args, **kwargs)
            # if we already have a call to the function currently waiting to be executed, reset the timer
            if debounced._timer is not None:
                debounced._timer.cancel()

            # after wait_time, call the function provided to the decorator with its arguments
            debounced._timer = threading.Timer(wait_time, call_function)
            debounced._timer.start()

        debounced._timer = None
        return debounced

    return decorator

def DREAMTEXTURES_HT_viewport_enabled(self, context):
    self.layout.prop(context.scene, "dream_textures_viewport_enabled", text="", icon="OUTLINER_OB_VOLUME" if context.scene.dream_textures_viewport_enabled else "VOLUME_DATA", toggle=True)

is_rendering_viewport = False
last_viewport_update = time.time()
last_viewport_pixel_buffer_update = time.time()
dream_viewport = None
is_rendering_dream = False
render_dream_flag = False
viewport_pixel_buffer = None
viewport_size = (0, 0)
ignore_next = 0
def create_image():
    print("Create image")
    global dream_viewport
    dream_viewport = bpy.data.images.new('Dream Viewport', width=32, height=32)

def register_render_pass():
    bpy.app.timers.register(create_image)

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

    def view_update_decorator(original):
        def view_update(self, context, depsgraph):
            result = original(self, context, depsgraph)
            global last_viewport_update
            global ignore_next
            if ignore_next <= 0:
                last_viewport_update = time.time()
                print("View Update")
            ignore_next -= 1
            return result
        return view_update
    cycles.CyclesRender.view_update = view_update_decorator(cycles.CyclesRender.view_update)
    
    def updates_stopped():
        global last_viewport_update
        global is_rendering_viewport
        global is_rendering_dream
        threshold_reached = (time.time() - last_viewport_update) < 0.5
        if threshold_reached != is_rendering_viewport:
            is_rendering_viewport = threshold_reached
            global viewport_pixel_buffer
            if not is_rendering_viewport and not is_rendering_dream and viewport_pixel_buffer is not None:
                print("Stopped rendering viewport")
                is_rendering_dream = True
                array = np.flipud((np.array(viewport_pixel_buffer) * 255).astype(np.int8))
                pixels_memory = SharedMemory(create=True, size=array.nbytes)
                pixels_memory_array = np.ndarray(array.shape, dtype=array.dtype, buffer=pixels_memory.buf)
                pixels_memory_array[:] = array[:]

                def image_callback(shared_memory_name, seed, width, height, upscaled=False):
                    if not upscaled:
                        shared_memory = SharedMemory(shared_memory_name)
                        pixels = np.frombuffer(shared_memory.buf, dtype=np.float32).copy()

                        global ignore_next
                        ignore_next = 5
                        global dream_viewport
                        dream_viewport.scale(width, height)
                        dream_viewport.pixels[:] = pixels

                        shared_memory.close()
                        pixels_memory.close()

                        print("Done")
                        global is_rendering_dream
                        is_rendering_dream = False
                        # for area in bpy.context.screen.areas:
                        #     if area.type == 'VIEW_3D':
                                # area.tag_redraw()
                
                def step_callback(step, width=None, height=None, shared_memory_name=None):
                    pass

                dream_texture(bpy.context.scene.dream_textures_render_properties_prompt, step_callback, image_callback, init_img_shared_memory=pixels_memory.name, init_img_shared_memory_width=viewport_size[0], init_img_shared_memory_height=viewport_size[1])
        return 0.5
    bpy.app.timers.register(updates_stopped)

    def draw():
        global last_viewport_pixel_buffer_update
        if not bpy.context.scene.dream_textures_viewport_enabled:
            return
        if (time.time() - last_viewport_pixel_buffer_update) < 0.5:
            return
        last_viewport_pixel_buffer_update = time.time()
        # get currently bound framebuffer
        framebuffer = gpu.state.active_framebuffer_get()

        # get information on current viewport 
        viewport_info = gpu.state.viewport_get()
        width = viewport_info[2]
        height = viewport_info[3]
        
        global viewport_pixel_buffer
        global viewport_size
        viewport_pixel_buffer = framebuffer.read_color(0, 0, width, height, 4, 0, 'FLOAT').to_list()
        viewport_size = (width, height)

    bpy.types.SpaceView3D.draw_handler_add(draw, (), 'WINDOW', 'PRE_VIEW')
    def draw_dream():
        global is_rendering_dream
        global is_rendering_viewport
        global dream_viewport
        if not bpy.context.scene.dream_textures_viewport_enabled or is_rendering_viewport:
            return
        texture = gpu.texture.from_image(dream_viewport)
        viewport_info = gpu.state.viewport_get()
        width = viewport_info[2]
        height = viewport_info[3]
        shader = gpu.shader.from_builtin("2D_IMAGE")
        shader.bind()
        shader.uniform_sampler("image", texture)
        batch = batch_for_shader(shader, 'TRI_FAN', {
            'pos': ((0, 0), (width, 0), (width, height), (0, height)),
            'texCoord': ((0, 0), (1, 0), (1, 1), (0, 1)),
        })
        batch.draw(shader)
    bpy.types.SpaceView3D.draw_handler_add(draw_dream, (), 'WINDOW', 'POST_PIXEL')

    bpy.types.VIEW3D_HT_header.append(DREAMTEXTURES_HT_viewport_enabled)

def unregister_render_pass():
    global update_render_passes_original
    cycles.CyclesRender.update_render_passes = update_render_passes_original
    global render_original
    cycles.CyclesRender.render = render_original
    # global del_original
    # cycles.CyclesRender.__del__ = del_original
    global view_update_original
    cycles.CyclesRender.view_update = view_update_original
    global view_draw_original
    cycles.CyclesRender.view_draw = view_draw_original
    
    bpy.types.VIEW3D_HT_header.remove(DREAMTEXTURES_HT_viewport_enabled)
