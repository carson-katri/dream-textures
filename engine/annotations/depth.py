import bpy
import gpu
from gpu_extras.batch import batch_for_shader
import numpy as np
import threading

def render_depth_map(context, collection=None, invert=True):
    e = threading.Event()
    result = None
    def _execute():
        nonlocal result
        width, height = context.scene.render.resolution_x, context.scene.render.resolution_y
        matrix = context.scene.camera.matrix_world.inverted()
        projection_matrix = context.scene.camera.calc_matrix_camera(
            context,
            x=width,
            y=height
        )
        offscreen = gpu.types.GPUOffScreen(width, height)

        with offscreen.bind():
            fb = gpu.state.active_framebuffer_get()
            fb.clear(color=(0.0, 0.0, 0.0, 0.0))
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.depth_mask_set(True)
            with gpu.matrix.push_pop():
                gpu.matrix.load_matrix(matrix)
                gpu.matrix.load_projection_matrix(projection_matrix)
                
                shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')

                for object in (context.scene.objects if collection is None else collection.objects):
                    object = object.evaluated_get(context)
                    try:
                        mesh = object.to_mesh(depsgraph=context).copy()
                    except:
                        continue
                    if mesh is None:
                        continue
                    vertices = np.empty((len(mesh.vertices), 3), 'f')
                    indices = np.empty((len(mesh.loop_triangles), 3), 'i')

                    mesh.transform(object.matrix_world)
                    mesh.vertices.foreach_get("co", np.reshape(vertices, len(mesh.vertices) * 3))
                    mesh.loop_triangles.foreach_get("vertices", np.reshape(indices, len(mesh.loop_triangles) * 3))
                    
                    batch = batch_for_shader(
                        shader, 'TRIS',
                        {"pos": vertices},
                        indices=indices,
                    )
                    batch.draw(shader)
            depth = np.array(fb.read_depth(0, 0, width, height).to_list())
            if invert:
                depth = 1 - depth
                mask = np.array(fb.read_color(0, 0, width, height, 4, 0, 'UBYTE').to_list())[:, :, 3]
                depth *= mask
            depth = np.interp(depth, [np.ma.masked_equal(depth, 0, copy=False).min(), depth.max()], [0, 1]).clip(0, 1)
        offscreen.free()
        result = depth
        e.set()
    bpy.app.timers.register(_execute, first_interval=0)
    e.wait()
    return result