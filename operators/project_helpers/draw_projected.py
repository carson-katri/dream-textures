import bpy
import gpu
from gpu_extras.batch import batch_for_shader
import numpy as np

def projected_shader():
    vert_out = gpu.types.GPUStageInterfaceInfo("my_interface")
    vert_out.smooth('VEC2', "uvInterp")

    shader_info = gpu.types.GPUShaderCreateInfo()
    shader_info.push_constant('MAT4', "ModelViewProjectionMatrix")
    shader_info.push_constant('VEC2', "RegionSize")
    shader_info.sampler(0, 'FLOAT_2D', "image")
    shader_info.vertex_in(0, 'VEC3', "pos")
    shader_info.vertex_out(vert_out)
    shader_info.fragment_out(0, 'VEC4', "fragColor")

    shader_info.vertex_source("""
void main()
{
  gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
  uvInterp = gl_Position.xy / RegionSize;

#ifdef USE_WORLD_CLIP_PLANES
  world_clip_planes_calc_clip_distance((clipPlanes.ClipModelMatrix * vec4(pos, 1.0)).xyz);
#endif
}
""")

    shader_info.fragment_source(
        "void main()"
        "{"
        "  fragColor = texture(image, gl_FragCoord.xy / RegionSize);"
        "}"
    )

    return gpu.shader.create_from_info(shader_info)

def draw_projected_map(context, matrix, projection_matrix, image):
    """
    Generate a facing map for the given matrices.
    """
    width, height = image.shape[:2]
    offscreen = gpu.types.GPUOffScreen(width, height)

    buffer = gpu.types.Buffer('FLOAT', width * height * 4, np.float16(image / 255).ravel())
    texture = gpu.types.GPUTexture(size=(width, height), data=buffer, format='RGB16F')

    with offscreen.bind():
        fb = gpu.state.active_framebuffer_get()
        fb.clear(color=(0.0, 0.0, 0.0, 0.0))
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(True)
        with gpu.matrix.push_pop():
            gpu.matrix.load_matrix(matrix)
            gpu.matrix.load_projection_matrix(projection_matrix)

            for ob in context.scene.objects:
                if (mesh := ob.data) is None:
                    continue

                if not hasattr(mesh, "transform"):
                    continue
                
                mesh = mesh.copy()

                mesh.transform(ob.matrix_world)
                mesh.calc_loop_triangles()

                co = np.empty((len(mesh.vertices), 3), 'f')
                normals = np.empty((len(mesh.vertices), 3), 'f')
                vertices = np.empty((len(mesh.loop_triangles), 3), 'i')

                mesh.vertices.foreach_get("co", co.ravel())
                mesh.vertices.foreach_get("normal", normals.ravel())
                mesh.loop_triangles.foreach_get("vertices", vertices.ravel())

                shader = projected_shader()
                batch = batch_for_shader(
                    shader, 'TRIS',
                    {"pos": co},
                    indices=vertices,
                )
                shader.uniform_float("RegionSize", (width, height))
                shader.uniform_sampler("image", texture)
                batch.draw(shader)
        projected = np.array(fb.read_color(0, 0, width, height, 4, 0, 'FLOAT').to_list())
    offscreen.free()
    return projected