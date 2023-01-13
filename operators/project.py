import bpy
import gpu
import bmesh
import mathutils
import mathutils.geometry
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
import numpy as np

from .project_helpers.draw_projected import draw_projected_map, blend_projections

from .view_history import ImportPromptFile
from ..property_groups.dream_prompt import pipeline_options
from .open_latest_version import OpenLatestVersion, is_force_show_download, new_version_available

from ..ui.panels.dream_texture import advanced_panel, create_panel, prompt_panel, size_panel
from .dream_texture import CancelGenerator, ReleaseGenerator
from ..preferences import StableDiffusionPreferences

from ..generator_process import Generator
from ..generator_process.actions.prompt_to_image import Pipeline
from ..generator_process.actions.huggingface_hub import ModelType

framebuffer_arguments = [
    ('depth', 'Depth', 'Only provide the scene depth as input'),
    ('color', 'Depth and Color', 'Provide the scene depth and color as input'),
]

class AddPerspective(bpy.types.Operator):
    bl_idname = "shade.dream_texture_project_add_perspective"
    bl_label = "Add Perspective"
    bl_description = "Adds the current view to the list of perspectives"
    bl_options = {'REGISTER'}

    def execute(self, context):
        perspective = context.scene.dream_textures_project_perspectives.add()
        perspective.name = f"Perspective {len(context.scene.dream_textures_project_perspectives)}"
        perspective.matrix = [c for v in context.space_data.region_3d.view_matrix for c in v]
        perspective.projection_matrix = [c for v in context.space_data.region_3d.window_matrix for c in v]
        return {'FINISHED'}

class RemovePerspective(bpy.types.Operator):
    bl_idname = "shade.dream_texture_project_remove_perspective"
    bl_label = "Remove Perspective"
    bl_description = "Removes a perspective"
    bl_options = {'REGISTER'}

    def execute(self, context):
        context.scene.dream_textures_project_perspectives.remove(context.scene.dream_textures_project_active_perspective)
        return {'FINISHED'}

class LoadPerspective(bpy.types.Operator):
    bl_idname = "shade.dream_texture_project_load_perspective"
    bl_label = "Load Perspective"
    bl_description = "Moves the viewport to the specified perspective"
    bl_options = {'REGISTER'}

    matrix: bpy.props.FloatVectorProperty(name="", size=4*4)
    projection_matrix: bpy.props.FloatVectorProperty(name="", size=4*4)

    def execute(self, context):
        context.space_data.region_3d.view_matrix = mathutils.Matrix([
            mathutils.Vector(self.matrix[i:i + 4])
            for i in range(0, len(self.matrix), 4)
        ])
        # context.space_data.region_3d.window_matrix = mathutils.Matrix([
        #     mathutils.Vector(self.projection_matrix[i:i + 4])
        #     for i in range(0, len(self.projection_matrix), 4)
        # ])
        for i in range(len(context.scene.dream_textures_project_perspectives)):
            if mathutils.Vector(context.scene.dream_textures_project_perspectives[i].matrix) == mathutils.Vector(self.matrix) \
                and mathutils.Vector(context.scene.dream_textures_project_perspectives[i].projection_matrix) == mathutils.Vector(self.projection_matrix):
                context.scene.dream_textures_project_active_perspective = i
        return {'FINISHED'}

class SCENE_UL_ProjectPerspectiveList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        layout.prop(item, "name", text="", emboss=False)
        active_perspective = mathutils.Vector([c for v in context.space_data.region_3d.view_matrix for c in v])
        is_same_perspective = (mathutils.Vector(item.matrix) - active_perspective).length < 0.00001
        load = layout.operator(LoadPerspective.bl_idname, text="", icon="RESTRICT_VIEW_OFF" if is_same_perspective else "RESTRICT_VIEW_ON")
        load.matrix = item.matrix
        load.projection_matrix = item.projection_matrix

def dream_texture_projection_panels():
    class DREAM_PT_dream_panel_projection(bpy.types.Panel):
        """Creates a Dream Textures panel for projection"""
        bl_label = "Dream Texture Projection"
        bl_idname = f"DREAM_PT_dream_panel_projection"
        bl_category = "Dream"
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'

        @classmethod
        def poll(cls, context):
            if cls.bl_space_type == 'NODE_EDITOR':
                return context.area.ui_type == "ShaderNodeTree" or context.area.ui_type == "CompositorNodeTree"
            else:
                return True
        
        def draw_header_preset(self, context):
            layout = self.layout
            layout.operator(ImportPromptFile.bl_idname, text="", icon="IMPORT")
            layout.separator()

        def draw(self, context):
            layout = self.layout
            layout.use_property_split = True
            layout.use_property_decorate = False

            if len(pipeline_options(self, context)) > 1:
                layout.prop(context.scene.dream_textures_project_prompt, "pipeline")
            if Pipeline[context.scene.dream_textures_project_prompt.pipeline].model():
                layout.prop(context.scene.dream_textures_project_prompt, 'model')
            
            if not Pipeline[context.scene.dream_textures_project_prompt.pipeline].depth():
                box = layout.box()
                box.label(text="Unsupported pipeline", icon="ERROR")
                box.label(text="The selected pipeline does not support depth to image.")
            
            models = list(filter(
                lambda m: m.model == context.scene.dream_textures_project_prompt.model,
                context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences.installed_models
            ))
            if len(models) > 0 and ModelType[models[0].model_type] != ModelType.DEPTH:
                box = layout.box()
                box.label(text="Unsupported model", icon="ERROR")
                box.label(text="Select a depth model, such as 'stabilityai/stable-diffusion-2-depth'")

            if is_force_show_download():
                layout.operator(OpenLatestVersion.bl_idname, icon="IMPORT", text="Download Latest Release")
            elif new_version_available():
                layout.operator(OpenLatestVersion.bl_idname, icon="IMPORT")

    yield DREAM_PT_dream_panel_projection

    def get_prompt(context):
        return context.scene.dream_textures_project_prompt
    yield from create_panel('VIEW_3D', 'UI', DREAM_PT_dream_panel_projection.bl_idname, prompt_panel, get_prompt)
    yield create_panel('VIEW_3D', 'UI', DREAM_PT_dream_panel_projection.bl_idname, size_panel, get_prompt)
    yield from create_panel('VIEW_3D', 'UI', DREAM_PT_dream_panel_projection.bl_idname, advanced_panel, get_prompt)

    def perspectives_panel(sub_panel, space_type, get_prompt):
        class PerspectivesPanel(sub_panel):
            bl_idname = f"DREAM_PT_dream_panel_projection_perspectives"
            bl_label = "Perspectives"

            def draw(self, context):
                layout = self.layout
                layout.use_property_split = True

                row = layout.row()
                row.template_list(SCENE_UL_ProjectPerspectiveList.__name__, "dream_textures_project_perspectives", context.scene, "dream_textures_project_perspectives", context.scene, "dream_textures_project_active_perspective")
                col = row.column(align=True)
                col.operator(AddPerspective.bl_idname, text="", icon="ADD")
                col.operator(RemovePerspective.bl_idname, text="", icon="REMOVE")
        return PerspectivesPanel

    yield create_panel('VIEW_3D', 'UI', DREAM_PT_dream_panel_projection.bl_idname, perspectives_panel, get_prompt)

    def actions_panel(sub_panel, space_type, get_prompt):
        class ActionsPanel(sub_panel):
            """Create a subpanel for actions"""
            bl_idname = f"DREAM_PT_dream_panel_projection_actions"
            bl_label = "Actions"
            bl_options = {'HIDE_HEADER'}

            def draw(self, context):
                super().draw(context)
                layout = self.layout
                layout.use_property_split = True

                layout.prop(context.scene, "dream_textures_project_framebuffer_arguments")
                if context.scene.dream_textures_project_framebuffer_arguments == 'color':
                    layout.prop(get_prompt(context), "strength")

                row = layout.row()
                row.scale_y = 1.5
                if context.scene.dream_textures_progress <= 0:
                    if context.scene.dream_textures_info != "":
                        row.label(text=context.scene.dream_textures_info, icon="INFO")
                    else:
                        r = row.row()
                        r.operator(ProjectDreamTexture.bl_idname, icon="MOD_UVPROJECT")
                        r.enabled = Pipeline[context.scene.dream_textures_project_prompt.pipeline].depth() and bpy.context.object.mode == 'EDIT'
                        if bpy.context.object.mode != 'EDIT':
                            box = layout.box()
                            box.label(text="Enter Edit Mode", icon="ERROR")
                            box.label(text="In edit mode, select the faces to project onto.")
                else:
                    disabled_row = row.row()
                    disabled_row.use_property_split = True
                    disabled_row.prop(context.scene, 'dream_textures_progress', slider=True)
                    disabled_row.enabled = False
                if CancelGenerator.poll(context):
                    row.operator(CancelGenerator.bl_idname, icon="CANCEL", text="")
                row.operator(ReleaseGenerator.bl_idname, icon="X", text="")
        return ActionsPanel
    yield create_panel('VIEW_3D', 'UI', DREAM_PT_dream_panel_projection.bl_idname, actions_panel, get_prompt)

def position_shader():
    vert_out = gpu.types.GPUStageInterfaceInfo("my_interface")
    vert_out.smooth('FLOAT', "facing")

    shader_info = gpu.types.GPUShaderCreateInfo()
    shader_info.push_constant('MAT4', "ModelViewProjectionMatrix")
    shader_info.vertex_in(0, 'VEC3', "pos")
    shader_info.vertex_in(1, 'VEC3', "normal")
    shader_info.vertex_out(vert_out)
    shader_info.fragment_out(0, 'VEC4', "fragColor")

    shader_info.vertex_source("""
void main()
{
  gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);

  vec3 clipSpaceViewDir = -normalize(vec3(ModelViewProjectionMatrix * vec4(0.0, 0.0, 0.0, 1.0)));
  vec3 viewDir = normalize((inverse(ModelViewProjectionMatrix) * vec4(clipSpaceViewDir, 0.0)).xyz);
  facing = dot(viewDir, normalize(normal));

#ifdef USE_WORLD_CLIP_PLANES
  world_clip_planes_calc_clip_distance((clipPlanes.ClipModelMatrix * vec4(pos, 1.0)).xyz);
#endif
}
""")

    shader_info.fragment_source(
        "void main()"
        "{"
        "  fragColor = vec4(facing, 0, 0, 1.0);"
        "}"
    )

    return gpu.shader.create_from_info(shader_info)

def draw_depth_map(width, height, context, matrix, projection_matrix):
    """
    Generate a depth map for the given matrices.
    """
    offscreen = gpu.types.GPUOffScreen(width, height)

    with offscreen.bind():
        fb = gpu.state.active_framebuffer_get()
        fb.clear(color=(0.0, 0.0, 0.0, 0.0))
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(True)
        with gpu.matrix.push_pop():
            gpu.matrix.load_matrix(matrix)
            gpu.matrix.load_projection_matrix(projection_matrix)
            
            offscreen.draw_view3d(
                context.scene,
                context.view_layer,
                context.space_data,
                context.region,
                matrix,
                projection_matrix,
                do_color_management=False
            )
        depth = np.array(fb.read_depth(0, 0, width, height).to_list())
        depth = 1 - depth
        depth = np.interp(depth, [np.ma.masked_equal(depth, 0, copy=False).min(), depth.max()], [0, 1]).clip(0, 1)
    offscreen.free()
    return depth

def draw_facing_map(width, height, context, matrix, projection_matrix):
    """
    Generate a facing map for the given matrices.
    """
    offscreen = gpu.types.GPUOffScreen(width, height)

    with offscreen.bind():
        fb = gpu.state.active_framebuffer_get()
        fb.clear(color=(0.0, 0.0, 0.0, 0.0))
        depth_test_setting = gpu.state.depth_test_get()
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

                shader = position_shader()
                batch = batch_for_shader(
                    shader, 'TRIS',
                    {"pos": co, "normal": normals},
                    indices=vertices,
                )
                batch.draw(shader)
        facing = np.array(fb.read_color(0, 0, width, height, 4, 0, 'FLOAT').to_list())
        gpu.state.depth_test_set(depth_test_setting)
    offscreen.free()
    return facing

#region view3d_utils
def region_2d_to_origin_3d(width, height, view_matrix, projection_matrix, coord, *, clamp=None, is_perspective=True):
    """
    Modified from https://github.com/blender/blender/blob/master/release/scripts/modules/bpy_extras/view3d_utils.py
    to take a `view_matrix` and `projection_matrix` instead of region data.
    """
    viewinv = view_matrix.inverted()
    perspective_matrix = projection_matrix * viewinv

    if is_perspective:
        origin_start = viewinv.translation.copy()
    else:
        persmat = perspective_matrix.copy()
        dx = (2.0 * coord[0] / width) - 1.0
        dy = (2.0 * coord[1] / height) - 1.0
        persinv = persmat.inverted()
        origin_start = (
            (persinv.col[0].xyz * dx) +
            (persinv.col[1].xyz * dy) +
            persinv.translation
        )

        if clamp != 0.0:
            # if view_perspective != 'CAMERA':
            if True: # We only care about the viewport perspective.
                # this value is scaled to the far clip already
                origin_offset = persinv.col[2].xyz
                if clamp is not None:
                    if clamp < 0.0:
                        origin_offset.negate()
                        clamp = -clamp
                    if origin_offset.length > clamp:
                        origin_offset.length = clamp

                origin_start -= origin_offset

    return origin_start

def region_2d_to_vector_3d(width, height, view_matrix, projection_matrix, coord, is_perspective=True):
    """
    Modified from https://github.com/blender/blender/blob/master/release/scripts/modules/bpy_extras/view3d_utils.py
    to take a `view_matrix` and `projection_matrix` instead of region data.
    """
    viewinv = view_matrix.inverted()
    perspective_matrix = projection_matrix * viewinv
    if is_perspective:
        persinv = perspective_matrix.inverted_safe()

        out = mathutils.Vector((
            (2.0 * coord[0] / width) - 1.0,
            (2.0 * coord[1] / height) - 1.0,
            -0.5
        ))

        w = out.dot(persinv[3].xyz) + persinv[3][3]

        view_vector = ((persinv @ out) / w) - viewinv.translation
    else:
        view_vector = -viewinv.col[2].xyz

    view_vector.normalize()

    return view_vector

def region_2d_to_location_3d(width, height, view_matrix, projection_matrix, coord, depth_location, is_perspective=True):
    """
    Modified from https://github.com/blender/blender/blob/master/release/scripts/modules/bpy_extras/view3d_utils.py
    to take a `view_matrix` and `projection_matrix` instead of region data.
    """
    coord_vec = region_2d_to_vector_3d(width, height, view_matrix, projection_matrix, coord, is_perspective)
    depth_location = mathutils.Vector(depth_location)

    origin_start = region_2d_to_origin_3d(width, height, view_matrix, projection_matrix, coord, is_perspective=is_perspective)
    origin_end = origin_start + coord_vec

    if is_perspective:
        viewinv = view_matrix.inverted()
        view_vec = viewinv.col[2].copy()
        return mathutils.geometry.intersect_line_plane(
            origin_start,
            origin_end,
            depth_location,
            view_vec, 1,
        )
    else:
        return mathutils.geometry.intersect_point_line(
            depth_location,
            origin_start,
            origin_end,
        )[0]

def location_3d_to_region_2d(width, height, view_matrix, projection_matrix, coord, *, default=None):
    """
    Modified from https://github.com/blender/blender/blob/master/release/scripts/modules/bpy_extras/view3d_utils.py
    to take a `view_matrix` and `projection_matrix` instead of region data.
    """
    perspective_matrix = projection_matrix * view_matrix.inverted()
    prj = perspective_matrix @ mathutils.Vector((coord[0], coord[1], coord[2], 1.0))
    if prj.w > 0.0:
        width_half = width / 2.0
        height_half = height / 2.0

        return mathutils.Vector((
            width_half + width_half * (prj.x / prj.w),
            height_half + height_half * (prj.y / prj.w),
        ))
    else:
        return default
#endregion

def coordinate_mapping(
    view_matrix1,
    projection_matrix1,
    depth1,
    view_matrix2,
    projection_matrix2,
    depth2
):
    # Generate a grid of pixel indices for the first depth map
    x, y = np.meshgrid(np.arange(depth1.shape[1]), np.arange(depth1.shape[0]))

    # Calculate the 3D positions of all the pixels in the first depth map using the depth values, view matrix, and window matrix
    positions_3d = np.einsum("ij,xyz->xyzj", np.linalg.inv(view_matrix1), np.stack((x, y, depth1, np.ones_like(depth1)), axis=-1))

    # Project the 3D positions onto the second depth map using the second view matrix and window matrix
    positions_2d = np.einsum("ij,xyzj->xyzj", view_matrix2, positions_3d)
    positions_2d = positions_2d / positions_2d[:, :, 3, np.newaxis]
    
    positions_2d = np.einsum("ij,xyzj->xyzj", projection_matrix2, positions_2d)

    # Round the 2D positions to the nearest pixel
    return np.round(positions_2d[:, :, :2]).astype(int)

class ProjectDreamTexture(bpy.types.Operator):
    bl_idname = "shade.dream_texture_project"
    bl_label = "Project Dream Texture"
    bl_description = "Automatically texture all selected objects using the depth buffer and Stable Diffusion"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return Generator.shared().can_use()

    @classmethod
    def get_uv_layer(cls, mesh: bmesh.types.BMesh):
        for i in range(len(mesh.loops.layers.uv)):
            uv = mesh.loops.layers.uv[i]
            if uv.name.lower() == "projected uvs":
                return uv
            
        return mesh.loops.layers.uv.new("Projected UVs")

    def execute(self, context):
        # Get region size
        region_width = region_height = None
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        region_width, region_height = region.width, region.height

        if region_width is None or region_height is None:
            self.report({'ERROR'}, "Could not determine region size.")

        # Capture depth
        depth_maps = []
        facing_maps = []
        for perspective in context.scene.dream_textures_project_perspectives:
            depth = Generator.shared().ndimage_zoom(
                draw_depth_map(
                    region_width,
                    region_height,
                    context,
                    mathutils.Matrix([
                        mathutils.Vector(perspective.matrix[i:i + 4])
                        for i in range(0, len(perspective.matrix), 4)
                    ]),
                    mathutils.Matrix([
                        mathutils.Vector(perspective.projection_matrix[i:i + 4])
                        for i in range(0, len(perspective.projection_matrix), 4)
                    ])
                ),
                (context.scene.dream_textures_project_prompt.width, context.scene.dream_textures_project_prompt.height),
            ).result()
            depth = np.uint8(depth * 255).astype(np.float32) / 255.
            depth_maps.append(depth)
            facing = Generator.shared().ndimage_zoom(
                draw_facing_map(
                    region_width,
                    region_height,
                    context,
                    mathutils.Matrix([
                        mathutils.Vector(perspective.matrix[i:i + 4])
                        for i in range(0, len(perspective.matrix), 4)
                    ]),
                    mathutils.Matrix([
                        mathutils.Vector(perspective.projection_matrix[i:i + 4])
                        for i in range(0, len(perspective.projection_matrix), 4)
                    ])
                ),
                (context.scene.dream_textures_project_prompt.width, context.scene.dream_textures_project_prompt.height, 4),
            ).result()
            facing = np.uint8(facing * 255).astype(np.float32) / 255.
            facing_maps.append(facing)

        # Create material
        material = bpy.data.materials.new(name="diffused-material")
        material.use_nodes = True
        image_texture_node = material.node_tree.nodes.new("ShaderNodeTexImage")
        material.node_tree.links.new(image_texture_node.outputs[0], material.node_tree.nodes['Principled BSDF'].inputs[0])
        uv_map_node = material.node_tree.nodes.new("ShaderNodeUVMap")
        uv_map_node.uv_map = "Projected UVs"
        material.node_tree.links.new(uv_map_node.outputs[0], image_texture_node.inputs[0])
        for obj in bpy.context.selected_objects:
            if not hasattr(obj, "data") or not hasattr(obj.data, "materials"):
                continue
            material_index = len(obj.material_slots)
            obj.data.materials.append(material)
            mesh = bmesh.from_edit_mesh(obj.data)
            # Project from UVs view and update material index
            mesh.verts.ensure_lookup_table()
            mesh.verts.index_update()
            def vert_to_uv(v):
                screen_space = view3d_utils.location_3d_to_region_2d(context.region, context.space_data.region_3d, obj.matrix_world @ v.co)
                if screen_space is None:
                    return None
                return (screen_space[0] / context.region.width, screen_space[1] / context.region.height)
            uv_layer = ProjectDreamTexture.get_uv_layer(mesh)
            mesh.faces.ensure_lookup_table()
            for face in mesh.faces:
                if face.select:
                    for loop in face.loops:
                        uv = vert_to_uv(mesh.verts[loop.vert.index])
                        if uv is None:
                            continue
                        loop[uv_layer].uv = uv
                    face.material_index = material_index
            bmesh.update_edit_mesh(obj.data)
        
        # Generate
        texture = None
        def on_response(_, response):
            nonlocal texture
            if texture is None:
                texture = bpy.data.images.new(name="Step", width=response.image.shape[1], height=response.image.shape[0])
            texture.name = f"Step {response.step}/{context.scene.dream_textures_project_prompt.steps}"
            texture.pixels[:] = response.image.ravel()
            texture.update()
            image_texture_node.image = texture

        def on_done(future):
            nonlocal texture
            generated = future.result()
            if isinstance(generated, list):
                generated = generated[-1]
            if texture is None:
                texture = bpy.data.images.new(name=str(generated.seed), width=generated.image.shape[1], height=generated.image.shape[0])
            texture.name = str(generated.seed)
            material.name = str(generated.seed)
            texture.pixels[:] = generated.image.ravel()
            texture.update()
        #     image_texture_node.image = texture

        step_groupings = [4, 23]
        prev_projected_maps = []
        for step_range in list(zip([None, *step_groupings[:-1]], step_groupings)):
            # Generate each perspective
            projected_maps = []
            for i, perspective in enumerate(context.scene.dream_textures_project_perspectives):
                print(f"Generating {perspective.name} at {step_range}")
                res = Generator.shared().depth_to_image(
                    depth=depth_maps[i],
                    image=prev_projected_maps[i] * 255 if len(prev_projected_maps) > 0 else None,
                    step_range=step_range,
                    **context.scene.dream_textures_project_prompt.generate_args()
                ).result()[-1].image
                projected_maps.append(res)
            prev_projected_maps.clear()
            # Blend each perspective together.
            print("Blending perspectives")
            for i, perspective in enumerate(context.scene.dream_textures_project_perspectives):
                for j, perspective_b in enumerate(context.scene.dream_textures_project_perspectives):
                    if i == j:
                        continue
                    matrix = mathutils.Matrix([
                        mathutils.Vector(perspective.matrix[i:i + 4])
                        for i in range(0, len(perspective.matrix), 4)
                    ])
                    projection_matrix = mathutils.Matrix([
                        mathutils.Vector(perspective.projection_matrix[i:i + 4])
                        for i in range(0, len(perspective.projection_matrix), 4)
                    ])
                    target_matrix = mathutils.Matrix([
                        mathutils.Vector(perspective_b.matrix[i:i + 4])
                        for i in range(0, len(perspective_b.matrix), 4)
                    ])
                    target_projection_matrix = mathutils.Matrix([
                        mathutils.Vector(perspective_b.projection_matrix[i:i + 4])
                        for i in range(0, len(perspective_b.projection_matrix), 4)
                    ])
                    # prev_projected_maps.append(projected_maps[i])
                    projected_b = draw_projected_map(
                        context,
                        matrix, projection_matrix,
                        target_matrix, target_projection_matrix,
                        projected_maps[j],
                    )
                    projected_facing_b = draw_projected_map(
                        context,
                        matrix, projection_matrix,
                        target_matrix, target_projection_matrix,
                        facing_maps[j],
                    )
                    prev_projected_maps.append(np.flipud(blend_projections(
                        projected_maps[i],
                        projected_b,
                        projected_facing_b
                    )))
            for i, map in enumerate(projected_maps):
                # if texture is None:
                texture = bpy.data.images.new(name="Step", width=map.shape[1], height=map.shape[0])
                texture.name = f"Step {step_range[0]} -> {step_range[1]} ({context.scene.dream_textures_project_perspectives[i].name})"
                texture.pixels[:] = map.ravel()
                texture.update()
            image_texture_node.image = texture

        print("Finished generating")

        return {'FINISHED'}
