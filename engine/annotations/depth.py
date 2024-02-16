import bpy
import gpu
from gpu_extras.batch import batch_for_shader
import numpy as np
import threading
from .compat import UNIFORM_COLOR

def render_depth_map(context, collection=None, invert=True, width=None, height=None, matrix=None, projection_matrix=None, main_thread=False):
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
                
                shader = gpu.shader.from_builtin(UNIFORM_COLOR)

                def render_mesh(mesh, transform):
                    mesh.transform(transform)
                    mesh.calc_loop_triangles()
                    vertices = np.empty((len(mesh.vertices), 3), 'f')
                    indices = np.empty((len(mesh.loop_triangles), 3), 'i')

                    mesh.vertices.foreach_get("co", np.reshape(vertices, len(mesh.vertices) * 3))
                    mesh.loop_triangles.foreach_get("vertices", np.reshape(indices, len(mesh.loop_triangles) * 3))
                    
                    batch = batch_for_shader(
                        shader, 'TRIS',
                        {"pos": vertices},
                        indices=indices,
                    )
                    batch.draw(shader)
                if collection is None:
                    for object in context.object_instances:
                        try:
                            mesh = object.object.to_mesh()
                            if mesh is not None:
                                render_mesh(mesh, object.matrix_world)
                                object.object.to_mesh_clear()
                        except:
                            continue
                else:
                    for object in collection.objects:
                        try:
                            mesh = object.to_mesh(depsgraph=context)
                            if mesh is not None:
                                render_mesh(mesh, object.matrix_world)
                                object.to_mesh_clear()
                        except:
                            continue
            depth = np.array(fb.read_depth(0, 0, width, height).to_list())
            if invert:
                depth = 1 - depth
                mask = np.array(fb.read_color(0, 0, width, height, 4, 0, 'UBYTE').to_list())[:, :, 3]
                depth *= mask
            depth = np.interp(depth, [np.ma.masked_equal(depth, 0, copy=False).min(), depth.max()], [0, 1]).clip(0, 1)
        gpu.state.depth_test_set('NONE')
        offscreen.free()
        result = depth
        e.set()
    if main_thread or threading.current_thread() == threading.main_thread():
        _execute()
        return result
    else:
        bpy.app.timers.register(_execute, first_interval=0)
        e.wait()
        return result