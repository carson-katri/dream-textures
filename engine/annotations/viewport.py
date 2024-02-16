import bpy
import gpu
import numpy as np
import threading

def render_viewport_color(context, width=None, height=None, matrix=None, projection_matrix=None, main_thread=False):
    e = threading.Event()
    result = None

    width, height = width or context.scene.render.resolution_x, height or context.scene.render.resolution_y
    matrix = matrix or context.scene.camera.matrix_world.inverted()
    projection_matrix = projection_matrix or context.scene.camera.calc_matrix_camera(
        context,
        x=width,
        y=height
    )

    def _execute():
        nonlocal result
        offscreen = gpu.types.GPUOffScreen(width, height)

        with offscreen.bind():
            fb = gpu.state.active_framebuffer_get()
            fb.clear(color=(0.0, 0.0, 0.0, 0.0), depth=1)
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.depth_mask_set(True)
            with gpu.matrix.push_pop():
                gpu.matrix.load_matrix(matrix)
                gpu.matrix.load_projection_matrix(projection_matrix)
                area = next(a for a in bpy.context.screen.areas if a.type == 'VIEW_3D')
                offscreen.draw_view3d(
                    context.scene,
                    context.view_layer,
                    next(s for s in area.spaces),
                    next(r for r in area.regions if r.type == 'WINDOW'),
                    matrix,
                    projection_matrix,
                    do_color_management=False
                )
            color = np.array(fb.read_color(0, 0, width, height, 4, 0, 'FLOAT').to_list())
        gpu.state.depth_test_set('NONE')
        offscreen.free()
        result = color
        e.set()
    if main_thread or threading.current_thread() == threading.main_thread():
        _execute()
        return result
    else:
        bpy.app.timers.register(_execute, first_interval=0)
        e.wait()
        return result