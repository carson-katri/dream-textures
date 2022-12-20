# Realtime Viewport is still under development, and is not currently used.
import bpy
import cycles
import time
import threading
import gpu
from gpu_extras.batch import batch_for_shader
import numpy as np
from multiprocessing.shared_memory import SharedMemory
from .operators.dream_texture import dream_texture

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

def register_realtime_viewport():
    bpy.app.timers.register(create_image)

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

def unregister_realtime_viewport():
    global view_update_original
    cycles.CyclesRender.view_update = view_update_original
    global view_draw_original
    cycles.CyclesRender.view_draw = view_draw_original
    
    bpy.types.VIEW3D_HT_header.remove(DREAMTEXTURES_HT_viewport_enabled)