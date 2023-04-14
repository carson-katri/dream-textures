import bpy
import gpu
from gpu_extras.batch import batch_for_shader
import numpy as np
import threading

def render_normal_map(context, collection=None, width=None, height=None, matrix=None, projection_matrix=None, main_thread=False):
    e = threading.Event()
    result = None

    width, height = width or context.scene.render.resolution_x, height or context.scene.render.resolution_y
    matrix = matrix or context.scene.camera.matrix_world.inverted()
    projection_matrix = projection_matrix or context.scene.camera.calc_matrix_camera(
        context,
        x=width,
        y=height
    )

    def normals_shader():
        vert_out = gpu.types.GPUStageInterfaceInfo("my_interface")
        vert_out.smooth('VEC3', "color")

        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.push_constant('MAT4', "ModelViewProjectionMatrix")
        shader_info.push_constant('MAT4', "ModelViewMatrix")
        shader_info.vertex_in(0, 'VEC3', "position")
        shader_info.vertex_in(1, 'VEC3', "normal")
        shader_info.vertex_out(vert_out)
        shader_info.fragment_out(0, 'VEC4', "fragColor")

        shader_info.vertex_source("""
void main()
{
    gl_Position = ModelViewProjectionMatrix * vec4(position, 1.0f);
    color = normalize(mat3(ModelViewMatrix) * vec3(normal.y, normal.x, normal.z));
}
""")

        shader_info.fragment_source("""
void main()
{
    // fragColor = vec4(color.b, color.g, color.r, color.a);
    fragColor = vec4((color + 1) / 2, 1.0f);
}
""")

        return gpu.shader.create_from_info(shader_info)

    def _execute():
        nonlocal result
        offscreen = gpu.types.GPUOffScreen(width, height)

        with offscreen.bind():
            fb = gpu.state.active_framebuffer_get()
            fb.clear(color=(0.5, 0.5, 1.0, 1.0), depth=1)
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.depth_mask_set(True)
            with gpu.matrix.push_pop():
                gpu.matrix.load_matrix(matrix)
                gpu.matrix.load_projection_matrix(projection_matrix)
                
                shader = normals_shader()
                shader.uniform_float("ModelViewMatrix", gpu.matrix.get_model_view_matrix())
                shader.bind()

                def render_mesh(mesh, transform):
                    mesh.transform(transform)
                    mesh.calc_loop_triangles()
                    mesh.calc_normals()
                    vertices = np.empty((len(mesh.vertices), 3), 'f')
                    normals = np.empty((len(mesh.vertices), 3), 'f')
                    indices = np.empty((len(mesh.loop_triangles), 3), 'i')

                    mesh.vertices.foreach_get("co", np.reshape(vertices, len(mesh.vertices) * 3))
                    mesh.vertices.foreach_get("normal", np.reshape(normals, len(mesh.vertices) * 3))
                    mesh.loop_triangles.foreach_get("vertices", np.reshape(indices, len(mesh.loop_triangles) * 3))
                    
                    batch = batch_for_shader(
                        shader, 'TRIS',
                        {"position": vertices, "normal": normals},
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
            normal_map = np.array(fb.read_color(0, 0, width, height, 4, 0, 'FLOAT').to_list())
        gpu.state.depth_test_set('NONE')
        offscreen.free()
        result = normal_map
        e.set()
    if main_thread:
        _execute()
        return result
    else:
        bpy.app.timers.register(_execute, first_interval=0)
        e.wait()
        return result