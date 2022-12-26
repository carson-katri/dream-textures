import bpy
import gpu
import bmesh
from bpy_extras import view3d_utils
import numpy as np

from .view_history import ImportPromptFile
from ..property_groups.dream_prompt import pipeline_options
from .open_latest_version import OpenLatestVersion, is_force_show_download, new_version_available

from ..ui.panels.dream_texture import advanced_panel, create_panel, prompt_panel, size_panel
from .dream_texture import CancelGenerator, ReleaseGenerator
from ..preferences import StableDiffusionPreferences

from ..generator_process import Generator
from ..generator_process.actions.prompt_to_image import Pipeline
from ..generator_process.actions.huggingface_hub import ModelType
import tempfile

framebuffer_arguments = [
    ('depth', 'Depth', 'Only provide the scene depth as input'),
    ('color', 'Depth and Color', 'Provide the scene depth and color as input'),
]

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

def draw(context, init_img_path, image_texture_node, material, cleanup):
    try:
        context.scene.dream_textures_info = "Rendering viewport depth..."
        framebuffer = gpu.state.active_framebuffer_get()
        viewport = gpu.state.viewport_get()
        width, height = viewport[2], viewport[3]
        depth = np.array(framebuffer.read_depth(0, 0, width, height).to_list())

        depth = 1 - depth
        depth = np.interp(depth, [np.ma.masked_equal(depth, 0, copy=False).min(), depth.max()], [0, 1]).clip(0, 1)

        gen = Generator.shared()
        
        texture = None

        def on_response(_, response):
            nonlocal texture
            if response.final:
                return
            context.scene.dream_textures_progress = response.step
            if texture is None:
                texture = bpy.data.images.new(name="Step", width=response.image.shape[1], height=response.image.shape[0])
            texture.name = f"Step {response.step}/{context.scene.dream_textures_project_prompt.steps}"
            texture.pixels[:] = response.image.ravel()
            texture.update()
            image_texture_node.image = texture

        def on_done(future):
            nonlocal texture
            if hasattr(gen, '_active_generation_future'):
                del gen._active_generation_future
            context.scene.dream_textures_info = ""
            context.scene.dream_textures_progress = 0
            generated = future.result()
            if isinstance(generated, list):
                generated = generated[-1]
            if texture is None:
                texture = bpy.data.images.new(name=str(generated.seed), width=generated.image.shape[1], height=generated.image.shape[0])
            texture.name = str(generated.seed)
            material.name = str(generated.seed)
            texture.pixels[:] = generated.image.ravel()
            texture.update()
            image_texture_node.image = texture
        
        def on_exception(_, exception):
            context.scene.dream_textures_info = ""
            context.scene.dream_textures_progress = 0
            if hasattr(gen, '_active_generation_future'):
                del gen._active_generation_future
            raise exception
        
        context.scene.dream_textures_info = "Starting..."
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

    finally:
        cleanup()

class ProjectDreamTexture(bpy.types.Operator):
    bl_idname = "shade.dream_texture_project"
    bl_label = "Project Dream Texture"
    bl_description = "Automatically texture all selected objects using the depth buffer and Stable Diffusion"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return Generator.shared().can_use()

    @classmethod
    def get_uv_layer(cls, mesh:bmesh.types.BMesh):
        for i in range(len(mesh.loops.layers.uv)):
            uv = mesh.loops.layers.uv[i]
            if uv.name.lower() == "projected uvs":
                return uv
            
        return mesh.loops.layers.uv.new("Projected UVs")

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

        context.scene.dream_textures_info = "Requesting redraw..."
        handle = None
        handle = bpy.types.SpaceView3D.draw_handler_add(
            draw,
            (
                context,
                init_img_path,
                image_texture_node,
                material,
                lambda: bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW'),
            ),
            'WINDOW',
            'POST_VIEW'
        )

        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
                return {'FINISHED'}

        return {'FINISHED'}
