import bpy
import gpu
import gpu.texture
from gpu_extras.batch import batch_for_shader
import bmesh
from bpy_extras import view3d_utils
import mathutils
import numpy as np

from .view_history import ImportPromptFile
from ..property_groups.dream_prompt import pipeline_options
from .open_latest_version import OpenLatestVersion, is_force_show_download, new_version_available

from ..ui.panels.dream_texture import advanced_panel, create_panel, prompt_panel, size_panel
from .dream_texture import CancelGenerator, ReleaseGenerator
from .notify_result import NotifyResult
from ..preferences import StableDiffusionPreferences

from ..generator_process import Generator
from ..generator_process.models import Pipeline, FixItError
from ..generator_process.actions.huggingface_hub import ModelType
import tempfile

framebuffer_arguments = [
    ('depth', 'Depth', 'Only provide the scene depth as input'),
    ('color', 'Depth and Color', 'Provide the scene depth and color as input'),
]

def _validate_projection(context):
    if len(context.selected_objects) == 0:
        def fix_selection(context, layout):
            if context.object.mode != 'OBJECT':
                layout.operator("object.mode_set", text="Switch to Object Mode", icon="OBJECT_DATAMODE").mode = 'OBJECT'
            layout.operator("object.select_by_type", text="Select All Meshes", icon="RESTRICT_SELECT_OFF").type = 'MESH'
        raise FixItError(
            """No objects selected
Select at least one object to project onto.""",
            fix_selection
        )
    if context.object is not None and context.object.mode != 'EDIT':
        def fix_mode(_, layout):
            layout.operator("object.mode_set", text="Switch to Edit Mode", icon="EDITMODE_HLT").mode = 'EDIT'
        raise FixItError(
            """Enter edit mode
In edit mode, select the faces to project onto.""",
            fix_mode
        )
    has_selection = False
    for obj in context.selected_objects:
        if not hasattr(obj, "data"):
            continue
        mesh = bmesh.from_edit_mesh(obj.data)
        bm = mesh.copy()
        bm.select_mode = {'FACE'}
        for f in bm.faces:
            if f.select:
                has_selection = True
                break
    if not has_selection:
        raise FixItError(
            """No faces selected.
Select at least one face to project onto.""",
            lambda ctx, layout: None
        )

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
            
            if is_force_show_download():
                layout.operator(OpenLatestVersion.bl_idname, icon="IMPORT", text="Download Latest Release")
            elif new_version_available():
                layout.operator(OpenLatestVersion.bl_idname, icon="IMPORT")

            layout.prop(context.scene.dream_textures_project_prompt, "pipeline")
            if Pipeline[context.scene.dream_textures_project_prompt.pipeline].model():
                layout.prop(context.scene.dream_textures_project_prompt, 'model')

    yield DREAM_PT_dream_panel_projection

    def get_prompt(context):
        return context.scene.dream_textures_project_prompt
    yield from create_panel('VIEW_3D', 'UI', DREAM_PT_dream_panel_projection.bl_idname, prompt_panel, get_prompt)
    yield create_panel('VIEW_3D', 'UI', DREAM_PT_dream_panel_projection.bl_idname, size_panel, get_prompt)
    yield from create_panel('VIEW_3D', 'UI', DREAM_PT_dream_panel_projection.bl_idname, advanced_panel, get_prompt)
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

                prompt = get_prompt(context)

                layout.prop(context.scene, "dream_textures_project_framebuffer_arguments")
                if context.scene.dream_textures_project_framebuffer_arguments == 'color':
                    layout.prop(prompt, "strength")
                
                col = layout.column()
                
                col.prop(context.scene, "dream_textures_project_use_control_net")
                if context.scene.dream_textures_project_use_control_net and len(prompt.control_nets) > 0:
                    col.prop(prompt.control_nets[0], "control_net", text="Depth ControlNet")
                    col.prop(prompt.control_nets[0], "conditioning_scale", text="ControlNet Conditioning Scale")

                col.prop(context.scene, "dream_textures_project_bake")
                if context.scene.dream_textures_project_bake:
                    for obj in context.selected_objects:
                        col.prop_search(obj.data.uv_layers, "active", obj.data, "uv_layers", text=f"{obj.name} Target UVs")

                row = layout.row()
                row.scale_y = 1.5
                if context.scene.dream_textures_progress <= 0:
                    if context.scene.dream_textures_info != "":
                        row.label(text=context.scene.dream_textures_info, icon="INFO")
                    else:
                        r = row.row()
                        r.operator(ProjectDreamTexture.bl_idname, icon="MOD_UVPROJECT")
                        r.enabled = Pipeline[context.scene.dream_textures_project_prompt.pipeline].depth() and context.object is not None and context.object.mode == 'EDIT'
                else:
                    disabled_row = row.row()
                    disabled_row.use_property_split = True
                    disabled_row.prop(context.scene, 'dream_textures_progress', slider=True)
                    disabled_row.enabled = False
                if CancelGenerator.poll(context):
                    row.operator(CancelGenerator.bl_idname, icon="CANCEL", text="")
                row.operator(ReleaseGenerator.bl_idname, icon="X", text="")
                
                # Validation
                try:
                    prompt.validate(context, task=None if context.scene.dream_textures_project_use_control_net else ModelType.DEPTH)
                    _validate_projection(context)
                except FixItError as e:
                    error_box = layout.box()
                    error_box.use_property_split = False
                    for i, line in enumerate(e.args[0].split('\n')):
                        error_box.label(text=line, icon="ERROR" if i == 0 else "NONE")
                    e.draw(context, error_box)
                except Exception as e:
                    print(e)
        return ActionsPanel
    yield create_panel('VIEW_3D', 'UI', DREAM_PT_dream_panel_projection.bl_idname, actions_panel, get_prompt)

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

def bake(context, mesh, width, height, src, projection_uvs, uvs):
    def bake_shader():
        vert_out = gpu.types.GPUStageInterfaceInfo("my_interface")
        vert_out.smooth('VEC2', "uvInterp")

        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.push_constant('MAT4', 'ModelViewProjectionMatrix')
        shader_info.sampler(0, 'FLOAT_2D', "image")
        shader_info.vertex_in(0, 'VEC2', "pos")
        shader_info.vertex_in(1, 'VEC2', "uv")
        shader_info.vertex_out(vert_out)
        shader_info.fragment_out(0, 'VEC4', "fragColor")

        shader_info.vertex_source("""
void main()
{
    gl_Position = ModelViewProjectionMatrix * vec4(pos * 2 - 1, 0.0, 1.0);
    uvInterp = uv;
}
""")

        shader_info.fragment_source("""
void main()
{
    fragColor = texture(image, uvInterp);
}
""")

        return gpu.shader.create_from_info(shader_info)

    offscreen = gpu.types.GPUOffScreen(width, height)

    buffer = gpu.types.Buffer('FLOAT', width * height * 4, src)
    texture = gpu.types.GPUTexture(size=(width, height), data=buffer, format='RGBA16F')
    
    with offscreen.bind():
        fb = gpu.state.active_framebuffer_get()
        fb.clear(color=(0.0, 0.0, 0.0, 0.0))
        with gpu.matrix.push_pop():
            gpu.matrix.load_matrix(mathutils.Matrix.Identity(4))
            gpu.matrix.load_projection_matrix(mathutils.Matrix.Identity(4))

            vertices = np.array([[l.vert.index for l in loop] for loop in mesh.calc_loop_triangles()], dtype='i')

            shader = bake_shader()
            batch = batch_for_shader(
                shader, 'TRIS',
                {"pos": uvs, "uv": projection_uvs},
                indices=vertices,
            )
            shader.uniform_sampler("image", texture)
            batch.draw(shader)
        projected = np.array(fb.read_color(0, 0, width, height, 4, 0, 'FLOAT').to_list())
    offscreen.free()
    return projected

def draw_color(width, height, context, mesh, matrix, projection_matrix, texture, uvs):
    """
    Generate a depth map for the given matrices.
    """
    offscreen = gpu.types.GPUOffScreen(width, height)
    buffer = gpu.types.Buffer('FLOAT', width * height * 4, texture)
    texture = gpu.types.GPUTexture(size=(width, height), data=buffer, format='RGBA16F')

    def color_shader():
        vert_out = gpu.types.GPUStageInterfaceInfo("my_interface")
        vert_out.smooth('VEC2', "uvInterp")

        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.push_constant('MAT4', 'ModelViewProjectionMatrix')
        shader_info.sampler(0, 'FLOAT_2D', "image")
        shader_info.vertex_in(0, 'VEC3', "pos")
        shader_info.vertex_in(1, 'VEC2', "uv")
        shader_info.vertex_out(vert_out)
        shader_info.fragment_out(0, 'VEC4', "fragColor")

        shader_info.vertex_source("""
void main()
{
    gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
    uvInterp = uv;
}
""")

        shader_info.fragment_source("""
void main()
{
    fragColor = texture(image, uvInterp);
}
""")

        return gpu.shader.create_from_info(shader_info)

    with offscreen.bind():
        fb = gpu.state.active_framebuffer_get()
        fb.clear(color=(0.0, 0.0, 0.0, 1.0))
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(True)
        with gpu.matrix.push_pop():
            gpu.matrix.load_matrix(matrix)
            gpu.matrix.load_projection_matrix(projection_matrix)

            mesh.calc_loop_triangles()

            vertices = np.array([vert.co for vert in mesh.verts], dtype='f')
            indices = np.array([[l.vert.index for l in loop] for loop in mesh.calc_loop_triangles()], dtype='i')
            
            shader = color_shader()
            batch = batch_for_shader(
                shader, 'TRIS',
                {"pos": vertices, "uv": uvs},
                indices=indices,
            )
            shader.uniform_sampler("image", texture)
            batch.draw(shader)
        color = np.array(fb.read_color(0, 0, width, height, 4, 0, 'FLOAT').to_list())
    offscreen.free()
    
    return color

def draw_mask_map(width, height, context, mesh, matrix, projection_matrix, threshold, bake):
    """
    Generate a depth map for the given matrices.
    """
    offscreen = gpu.types.GPUOffScreen(width, height)

    def mask_shader():
        vert_out = gpu.types.GPUStageInterfaceInfo("my_interface")
        vert_out.smooth('VEC3', "normal")

        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.push_constant('MAT4', 'ModelViewProjectionMatrix')
        shader_info.push_constant('MAT3', 'NormalMatrix')
        shader_info.push_constant('FLOAT', 'Threshold')
        shader_info.push_constant('VEC3', 'CameraNormal')
        shader_info.vertex_in(0, 'VEC3', "pos")
        shader_info.vertex_in(1, 'VEC3', "nor")
        shader_info.vertex_out(vert_out)
        shader_info.fragment_out(0, 'VEC4', "fragColor")

        shader_info.vertex_source("""
void main()
{
    gl_Position = ModelViewProjectionMatrix * vec4(pos, 1.0);
    normal = nor;
}
""")


        shader_info.fragment_source("""
void main()
{
    fragColor = dot(NormalMatrix * normal, CameraNormal) > Threshold ? vec4(1, 1, 1, 1) : vec4(0, 0, 0, 1);
}
""")

        return gpu.shader.create_from_info(shader_info)

    with offscreen.bind():
        fb = gpu.state.active_framebuffer_get()
        fb.clear(color=(0.0, 0.0, 0.0, 0.0))
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(True)
        with gpu.matrix.push_pop():
            if bake is not None:
                gpu.matrix.load_matrix(mathutils.Matrix.Identity(4))
                gpu.matrix.load_projection_matrix(mathutils.Matrix.Identity(4))
            else:
                gpu.matrix.load_matrix(matrix)
                gpu.matrix.load_projection_matrix(projection_matrix)

            mesh.calc_loop_triangles()

            vertices = np.array([vert.co for vert in mesh.verts], dtype='f')
            normals = np.array([vert.normal for vert in mesh.verts], dtype='f')
            indices = np.array([[l.vert.index for l in loop] for loop in mesh.calc_loop_triangles()], dtype='i')
            
            shader = mask_shader()
            shader.uniform_float('CameraNormal', context.region_data.view_rotation @ mathutils.Vector((0, 0, 1)) if bake is not None else (0, 0, 1))
            shader.uniform_float('Threshold', threshold)
            
            batch = batch_for_shader(
                shader, 'TRIS',
                {"pos": bake if bake is not None else vertices, "nor": normals},
                indices=indices,
            )
            batch.draw(shader)
        mask = np.array(fb.read_color(0, 0, width, height, 4, 0, 'FLOAT').to_list())
    offscreen.free()
    
    return mask

class ProjectDreamTexture(bpy.types.Operator):
    bl_idname = "shade.dream_texture_project"
    bl_label = "Project Dream Texture"
    bl_description = "Automatically texture all selected objects using the depth buffer and Stable Diffusion"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return True
        # try:
        #     context.scene.dream_textures_project_prompt.validate(context, task=None if context.scene.dream_textures_project_use_control_net else ModelType.DEPTH)
        #     _validate_projection(context)
        # except:
        #     return False
        # return Generator.shared().can_use()

    @classmethod
    def get_uv_layer(cls, mesh: bmesh.types.BMesh):
        for i in range(len(mesh.loops.layers.uv)):
            uv = mesh.loops.layers.uv[i]
            if uv.name.lower() == "projected uvs":
                return uv, i

        return mesh.loops.layers.uv.new("Projected UVs"), len(mesh.loops.layers.uv) - 1

    def draw_color(self, context):
        res_x, res_y = context.scene.render.resolution_x, context.scene.render.resolution_y
        view3d_spaces = []
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        context.scene.render.resolution_x, context.scene.render.resolution_y = region.width, region.height
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        if space.overlay.show_overlays:
                            view3d_spaces.append(space)
                            space.overlay.show_overlays = False
        init_img_path = tempfile.NamedTemporaryFile(suffix='.png').name
        render_filepath, file_format = context.scene.render.filepath, context.scene.render.image_settings.file_format
        context.scene.render.image_settings.file_format = 'PNG'
        context.scene.render.filepath = init_img_path
        bpy.ops.render.opengl(write_still=True, view_context=True)
        for space in view3d_spaces:
            space.overlay.show_overlays = True
        context.scene.render.resolution_x, context.scene.render.resolution_y = res_x, res_y
        context.scene.render.filepath, context.scene.render.image_settings.file_format = render_filepath, file_format
        return init_img_path

    def masked_init_image(self, context, camera, texture, mesh, split_mesh, uvs, threshold=0.5):
        mask = draw_mask_map(512, 512, context, mesh, camera.matrix_world.inverted(), context.space_data.region_3d.window_matrix, threshold, bake=None)
        color = draw_color(512, 512, context, split_mesh, camera.matrix_world.inverted(), context.space_data.region_3d.window_matrix, texture, uvs)
        bake_uvs = [(uv[0] * 2 - 1, uv[1] * 2 - 1, 0) for uv in uvs]
        baked_mask = draw_mask_map(512, 512, context, split_mesh, camera.matrix_world.inverted(), context.space_data.region_3d.window_matrix, threshold, bake=bake_uvs)

        return color, mask, baked_mask

    def execute(self, context):
        texture = bpy.data.images['result']
        generation_result = bpy.data.images['generation_result']
        inpaint_result = bpy.data.images['inpaint_result']
        mask_result = bpy.data.images['mask']

        gen = Generator.shared()

        step_i = 0
        def step(camera, inpaint):
            def location_3d_to_region_2d(coord):
                perspective_matrix = context.space_data.region_3d.window_matrix @ camera.matrix_world.inverted()
                prj = perspective_matrix @ mathutils.Vector((coord[0], coord[1], coord[2], 1.0))
                if prj.w > 0.0:
                    width_half = context.region.width / 2.0
                    height_half = context.region.height / 2.0

                    return mathutils.Vector((
                        width_half + width_half * (prj.x / prj.w),
                        height_half + height_half * (prj.y / prj.w),
                    ))
                else:
                    return None
                
            mesh = bmesh.from_edit_mesh(context.object.data)
            mesh.verts.ensure_lookup_table()
            mesh.verts.index_update()

            split_mesh = mesh.copy()
            split_mesh.transform(context.object.matrix_world)
            split_mesh.select_mode = {'FACE'}
            bmesh.ops.split_edges(split_mesh, edges=split_mesh.edges)
            
            split_mesh.faces.ensure_lookup_table()
            split_mesh.verts.ensure_lookup_table()
            split_mesh.verts.index_update()

            def vert_to_uv(v):
                screen_space = location_3d_to_region_2d(v.co)
                if screen_space is None:
                    return None
                return (screen_space[0] / context.region.width, screen_space[1] / context.region.height)

            uv_layer = split_mesh.loops.layers.uv.active
            uvs = np.empty((len(split_mesh.verts), 2), dtype=np.float32)
            projection_uvs = np.empty((len(split_mesh.verts), 2), dtype=np.float32)
            for face in split_mesh.faces:
                for loop in face.loops:
                    projection_uvs[loop.vert.index] = vert_to_uv(split_mesh.verts[loop.vert.index])
                    uvs[loop.vert.index] = loop[uv_layer].uv

            depth = np.flipud(
                draw_depth_map(512, 512, context, camera.matrix_world.inverted(), context.space_data.region_3d.window_matrix)
            )
            generated_args = context.scene.dream_textures_project_prompt.generate_args()
            del generated_args['model']
            del generated_args['control']
            del generated_args['inpaint_mask_src']

            if inpaint:
                color, mask, baked_mask = self.masked_init_image(context, camera, np.array(texture.pixels, dtype=np.float32), mesh, split_mesh, uvs)
                color[:, :, 3] = 1 - mask[:, :, 0]
                res = gen.control_net(
                    model='models--runwayml--stable-diffusion-inpainting',
                    control=[depth],
                    image=np.flipud(color * 255),
                    inpaint=True,
                    inpaint_mask_src='alpha',
                    **generated_args
                ).result()
                inpaint_result.pixels[:] = res[-1].images[0].ravel()
                inpaint_result.update()
                mask_result.pixels[:] = mask.ravel()
                mask_result.update()
                color = bake(context, split_mesh, 512, 512, res[-1].images[0].ravel(), projection_uvs, uvs).ravel()
                baked_mask = baked_mask.ravel()
                color = (np.array(texture.pixels, dtype=np.float32) * (1 - baked_mask)) + (color * baked_mask)
            else:
                res = gen.control_net(
                    model='models--runwayml--stable-diffusion-v1-5',
                    control=[depth],
                    image=None,
                    inpaint=False,
                    inpaint_mask_src='alpha',
                    **generated_args
                ).result()
                generation_result.pixels[:] = res[-1].images[0].ravel()
                generation_result.update()
                color = bake(context, split_mesh, 512, 512, res[-1].images[0].ravel(), projection_uvs, uvs)

            texture.pixels[:] = color.ravel()
            texture.update()
            nonlocal step_i
            bpy.data.images.new(name=f"Step {step_i}", width=512, height=512).pixels[:] = color.ravel()
            step_i += 1
        
        started = False
        for camera in [ob for ob in bpy.context.scene.objects if ob.type == 'CAMERA' and not ob.hide_viewport]:
            modelview_matrix = camera.matrix_world.inverted()
            context.space_data.region_3d.view_matrix = modelview_matrix
            step(camera, started)
            started = True

        return {'FINISHED'}

        # Setup the progress indicator
        def step_progress_update(self, context):
            if hasattr(context.area, "regions"):
                for region in context.area.regions:
                    if region.type == "UI":
                        region.tag_redraw()
            return None
        bpy.types.Scene.dream_textures_progress = bpy.props.IntProperty(name="", default=0, min=0, max=context.scene.dream_textures_project_prompt.steps, update=step_progress_update)
        context.scene.dream_textures_info = "Starting..."

        # Get region size
        region_width = region_height = None
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        region_width, region_height = region.width, region.height

        if region_width is None or region_height is None:
            self.report({'ERROR'}, "Could not determine region size.")

        # Render the viewport
        if context.scene.dream_textures_project_framebuffer_arguments == 'color':
            context.scene.dream_textures_info = "Rendering viewport color..."
            res_x, res_y = context.scene.render.resolution_x, context.scene.render.resolution_y
            view3d_spaces = []
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    for region in area.regions:
                        if region.type == 'WINDOW':
                            context.scene.render.resolution_x, context.scene.render.resolution_y = region.width, region.height
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            if space.overlay.show_overlays:
                                view3d_spaces.append(space)
                                space.overlay.show_overlays = False
            init_img_path = tempfile.NamedTemporaryFile(suffix='.png').name
            render_filepath, file_format = context.scene.render.filepath, context.scene.render.image_settings.file_format
            context.scene.render.image_settings.file_format = 'PNG'
            context.scene.render.filepath = init_img_path
            bpy.ops.render.opengl(write_still=True, view_context=True)
            for space in view3d_spaces:
                space.overlay.show_overlays = True
            context.scene.render.resolution_x, context.scene.render.resolution_y = res_x, res_y
            context.scene.render.filepath, context.scene.render.image_settings.file_format = render_filepath, file_format
        else:
            init_img_path = None

        context.scene.dream_textures_info = "Generating UVs and materials..."
        
        material = bpy.data.materials.new(name="diffused-material")
        material.use_nodes = True
        image_texture_node = material.node_tree.nodes.new("ShaderNodeTexImage")
        material.node_tree.links.new(image_texture_node.outputs[0], material.node_tree.nodes['Principled BSDF'].inputs[0])
        uv_map_node = material.node_tree.nodes.new("ShaderNodeUVMap")
        uv_map_node.uv_map = bpy.context.selected_objects[0].data.uv_layers.active.name if context.scene.dream_textures_project_bake else "Projected UVs"
        material.node_tree.links.new(uv_map_node.outputs[0], image_texture_node.inputs[0])
        target_objects = []
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
            uv_layer, uv_layer_index = ProjectDreamTexture.get_uv_layer(mesh)

            bm = mesh.copy()
            bm.select_mode = {'FACE'}
            bmesh.ops.split_edges(bm, edges=bm.edges)
            bmesh.ops.delete(bm, geom=[f for f in bm.faces if not f.select], context='FACES')
            target_objects.append((bm, bm.loops.layers.uv[uv_layer_index]))

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

        context.scene.dream_textures_info = "Rendering viewport depth..."

        depth = draw_depth_map(region_width, region_height, context, context.space_data.region_3d.view_matrix, context.space_data.region_3d.window_matrix)
        
        gen = Generator.shared()
        
        texture = None

        def on_response(_, response):
            nonlocal texture
            if response.final:
                return
            context.scene.dream_textures_progress = response.step
            if texture is None:
                texture = bpy.data.images.new(name="Step", width=response.images[0].shape[1], height=response.images[0].shape[0])
            texture.name = f"Step {response.step}/{context.scene.dream_textures_project_prompt.steps}"
            texture.pixels[:] = response.images[0].ravel()
            texture.update()
            image_texture_node.image = texture

        def on_done(future):
            nonlocal texture
            if hasattr(gen, '_active_generation_future'):
                del gen._active_generation_future
            context.scene.dream_textures_info = ""
            context.scene.dream_textures_progress = 0
            generated = future.result()
            prompt_subject = context.scene.dream_textures_project_prompt.prompt_structure_token_subject
            seed = generated[0].seeds[0]
            seed_str_length = len(str(seed))
            trim_aware_name = (prompt_subject[:54 - seed_str_length] + '..') if len(prompt_subject) > 54 else prompt_subject
            name_with_trimmed_prompt = f"{trim_aware_name} ({seed})"

            if isinstance(generated, list):
                generated = generated[-1]
            if texture is None:
                texture = bpy.data.images.new(name=name_with_trimmed_prompt, width=generated.images[0].shape[1], height=generated.images[0].shape[0])
            texture.name = name_with_trimmed_prompt
            material.name = name_with_trimmed_prompt
            texture.pixels[:] = generated.images[0].ravel()
            texture.update()
            texture.pack()
            image_texture_node.image = texture
            if context.scene.dream_textures_project_bake:
                for bm, src_uv_layer in target_objects:
                    dest = bpy.data.images.new(name=f"{texture.name} (Baked)", width=texture.size[0], height=texture.size[1])
                    
                    dest_uv_layer = bm.loops.layers.uv.active
                    src_uvs = np.empty((len(bm.verts), 2), dtype=np.float32)
                    dest_uvs = np.empty((len(bm.verts), 2), dtype=np.float32)
                    for face in bm.faces:
                        for loop in face.loops:
                            src_uvs[loop.vert.index] = loop[src_uv_layer].uv
                            dest_uvs[loop.vert.index] = loop[dest_uv_layer].uv
                    dest.pixels[:] = bake(context, bm, dest.size[0], dest.size[1], generated.images[0].ravel(), src_uvs, dest_uvs)
                    dest.update()
                    dest.pack()
                    image_texture_node.image = dest
        
        def on_exception(_, exception):
            context.scene.dream_textures_info = ""
            context.scene.dream_textures_progress = 0
            if hasattr(gen, '_active_generation_future'):
                del gen._active_generation_future
            eval('bpy.ops.' + NotifyResult.bl_idname)('INVOKE_DEFAULT', exception=repr(exception))
            raise exception
        
        context.scene.dream_textures_info = "Starting..."
        if context.scene.dream_textures_project_use_control_net:
            generated_args = context.scene.dream_textures_project_prompt.generate_args()
            del generated_args['control']
            future = gen.control_net(
                control=[np.flipud(depth)], # the depth control needs to be flipped.
                image=init_img_path,
                **generated_args
            )
        else:
            future = gen.depth_to_image(
                depth=depth,
                image=init_img_path,
                **context.scene.dream_textures_project_prompt.generate_args()
            )
        gen._active_generation_future = future
        future.call_done_on_exception = False
        future.add_response_callback(on_response)
        future.add_done_callback(on_done)
        future.add_exception_callback(on_exception)

        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
                return {'FINISHED'}

        return {'FINISHED'}
