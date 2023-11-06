import bpy
import gpu
import gpu.texture
from gpu_extras.batch import batch_for_shader
import bmesh
from bpy_extras import view3d_utils
import mathutils
import numpy as np
from typing import List

from .view_history import ImportPromptFile
from .open_latest_version import OpenLatestVersion, is_force_show_download, new_version_available

from ..ui.panels.dream_texture import advanced_panel, create_panel, prompt_panel, size_panel
from .dream_texture import CancelGenerator, ReleaseGenerator
from .notify_result import NotifyResult

from ..generator_process import Generator
from ..generator_process.models import ModelType
from ..api.models import FixItError
import tempfile

from ..engine.annotations.depth import render_depth_map

from .. import api

framebuffer_arguments = [
    ('depth', 'Depth', 'Only provide the scene depth as input'),
    ('color', 'Depth and Color', 'Provide the scene depth and color as input'),
]

def _validate_projection(context):
    if len(context.selected_objects) == 0:
        def object_mode_operator(operator):
            operator.mode = 'OBJECT'
        def select_by_type_operator(operator):
            operator.type = 'MESH'
        raise FixItError(
            """No objects selected
Select at least one object to project onto.""",
            FixItError.RunOperator("Switch to Object Mode", "object.mode_set", object_mode_operator)
            if context.object.mode != 'OBJECT'
            else FixItError.RunOperator("Select All Meshes", "object.select_by_type", select_by_type_operator)
        )
    if context.object is not None and context.object.mode != 'EDIT':
        def fix_mode(operator):
            operator.mode = 'EDIT'
        raise FixItError(
            """Enter edit mode
In edit mode, select the faces to project onto.""",
            FixItError.RunOperator("Switch to Edit Mode", "object.mode_set", fix_mode)
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
            FixItError.RunOperator("Select All Faces", "mesh.select_all", lambda _: None)
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

            layout.prop(context.scene.dream_textures_project_prompt, "backend")
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

                row = layout.row(align=True)
                row.scale_y = 1.5
                if CancelGenerator.poll(context):
                    row.operator(CancelGenerator.bl_idname, icon="SNAP_FACE", text="")
                if context.scene.dream_textures_progress <= 0:
                    if context.scene.dream_textures_info != "":
                        disabled_row = row.row(align=True)
                        disabled_row.operator(ProjectDreamTexture.bl_idname, text=context.scene.dream_textures_info, icon="INFO")
                        disabled_row.enabled = False
                    else:
                        r = row.row(align=True)
                        r.operator(ProjectDreamTexture.bl_idname, icon="MOD_UVPROJECT")
                        r.enabled = context.object is not None and context.object.mode == 'EDIT'
                else:
                    disabled_row = row.row(align=True)
                    disabled_row.use_property_split = True
                    disabled_row.prop(context.scene, 'dream_textures_progress', slider=True)
                    disabled_row.enabled = False
                row.operator(ReleaseGenerator.bl_idname, icon="X", text="")
                
                # Validation
                try:
                    _validate_projection(context)
                    prompt = context.scene.dream_textures_project_prompt
                    backend: api.Backend = prompt.get_backend()
                    args = prompt.generate_args(context)
                    args.task = api.task.PromptToImage() if context.scene.dream_textures_project_use_control_net else api.task.DepthToImage(None, None, 0)
                    backend.validate(args)
                except FixItError as e:
                    error_box = layout.box()
                    error_box.use_property_split = False
                    for i, line in enumerate(e.args[0].split('\n')):
                        error_box.label(text=line, icon="ERROR" if i == 0 else "NONE")
                    e._draw(context.scene.dream_textures_project_prompt, context, error_box)
                except Exception as e:
                    print(e)
        return ActionsPanel
    yield create_panel('VIEW_3D', 'UI', DREAM_PT_dream_panel_projection.bl_idname, actions_panel, get_prompt)

def bake(context, mesh, src, dest, src_uv, dest_uv):
    def bake_shader():
        vert_out = gpu.types.GPUStageInterfaceInfo("my_interface")
        vert_out.smooth('VEC2', "uvInterp")

        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.sampler(0, 'FLOAT_2D', "image")
        shader_info.vertex_in(0, 'VEC2', "src_uv")
        shader_info.vertex_in(1, 'VEC2', "dest_uv")
        shader_info.vertex_out(vert_out)
        shader_info.fragment_out(0, 'VEC4', "fragColor")

        shader_info.vertex_source("""
void main()
{
    gl_Position = vec4(dest_uv * 2 - 1, 0.0, 1.0);
    uvInterp = src_uv;
}
""")

        shader_info.fragment_source("""
void main()
{
    fragColor = texture(image, uvInterp);
}
""")

        return gpu.shader.create_from_info(shader_info)

    width, height = dest.size[0], dest.size[1]
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
                {"src_uv": src_uv, "dest_uv": dest_uv},
                indices=vertices,
            )
            shader.uniform_sampler("image", texture)
            batch.draw(shader)
        projected = np.array(fb.read_color(0, 0, width, height, 4, 0, 'FLOAT').to_list())
    offscreen.free()
    dest.pixels[:] = projected.ravel()

class ProjectDreamTexture(bpy.types.Operator):
    bl_idname = "shade.dream_texture_project"
    bl_label = "Project Dream Texture"
    bl_description = "Automatically texture all selected objects using the depth buffer and Stable Diffusion"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        try:
            _validate_projection(context)
            prompt = context.scene.dream_textures_project_prompt
            backend: api.Backend = prompt.get_backend()
            args = prompt.generate_args(context)
            args.task = api.task.PromptToImage() if context.scene.dream_textures_project_use_control_net else api.task.DepthToImage(None, None, 0)
            backend.validate(args)
        except:
            return False
        return Generator.shared().can_use()

    @classmethod
    def get_uv_layer(cls, mesh: bmesh.types.BMesh):
        for i in range(len(mesh.loops.layers.uv)):
            uv = mesh.loops.layers.uv[i]
            if uv.name.lower() == "projected uvs":
                return uv, i

        return mesh.loops.layers.uv.new("Projected UVs"), len(mesh.loops.layers.uv) - 1

    def execute(self, context):
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
        principled_node = next((n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'))
        material.node_tree.links.new(image_texture_node.outputs[0], principled_node.inputs[0])
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

        depth = render_depth_map(
            context.evaluated_depsgraph_get(),
            collection=None,
            width=region_width,
            height=region_height,
            matrix=context.space_data.region_3d.view_matrix,
            projection_matrix=context.space_data.region_3d.window_matrix,
            main_thread=True
        )
        
        texture = None

        def step_callback(progress: List[api.GenerationResult]) -> bool:
            nonlocal texture
            context.scene.dream_textures_progress = progress[-1].progress
            image = api.GenerationResult.tile_images(progress)
            if texture is None:
                texture = bpy.data.images.new(name="Step", width=image.shape[1], height=image.shape[0])
            texture.name = f"Step {progress[-1].progress}/{progress[-1].total}"
            texture.pixels[:] = image.ravel()
            texture.update()
            image_texture_node.image = texture
            return CancelGenerator.should_continue

        def callback(results: List[api.GenerationResult] | Exception):
            CancelGenerator.should_continue = None
            if isinstance(results, Exception):
                context.scene.dream_textures_info = ""
                context.scene.dream_textures_progress = 0
                if not isinstance(results, InterruptedError): # this is a user-initiated cancellation
                    eval('bpy.ops.' + NotifyResult.bl_idname)('INVOKE_DEFAULT', exception=repr(results))
                raise results
            else:
                nonlocal texture
                context.scene.dream_textures_info = ""
                context.scene.dream_textures_progress = 0
                result = results[-1]
                prompt_subject = context.scene.dream_textures_project_prompt.prompt_structure_token_subject
                seed_str_length = len(str(result.seed))
                trim_aware_name = (prompt_subject[:54 - seed_str_length] + '..') if len(prompt_subject) > 54 else prompt_subject
                name_with_trimmed_prompt = f"{trim_aware_name} ({result.seed})"

                if texture is None:
                    texture = bpy.data.images.new(name=name_with_trimmed_prompt, width=result.image.shape[1], height=result.image.shape[0])
                texture.name = name_with_trimmed_prompt
                material.name = name_with_trimmed_prompt
                texture.pixels[:] = result.image.ravel()
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
                        bake(context, bm, result.image.ravel(), dest, src_uvs, dest_uvs)
                        dest.update()
                        dest.pack()
                        image_texture_node.image = dest
        
        backend: api.Backend = context.scene.dream_textures_project_prompt.get_backend()

        context.scene.dream_textures_info = "Starting..."
        CancelGenerator.should_continue = True # reset global cancellation state
        image_data = bpy.data.images.load(init_img_path) if init_img_path is not None else None
        image = np.asarray(image_data.pixels).reshape((*depth.shape, image_data.channels)) if image_data is not None else None
        if context.scene.dream_textures_project_use_control_net:
            generated_args: api.GenerationArguments = context.scene.dream_textures_project_prompt.generate_args(context, init_image=image, control_images=[np.flipud(depth)])
            backend.generate(generated_args, step_callback=step_callback, callback=callback)
        else:
            generated_args: api.GenerationArguments = context.scene.dream_textures_project_prompt.generate_args(context)
            generated_args.task = api.DepthToImage(depth, image, context.scene.dream_textures_project_prompt.strength)
            backend.generate(generated_args, step_callback=step_callback, callback=callback)

        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
                return {'FINISHED'}

        return {'FINISHED'}
