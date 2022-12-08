import bpy
import gpu
import numpy as np
from mathutils import Matrix
from gpu_extras.batch import batch_for_shader

from .view_history import ImportPromptFile
from ..property_groups.dream_prompt import pipeline_options
from .open_latest_version import OpenLatestVersion, is_force_show_download, new_version_available

from ..ui.panels.dream_texture import advanced_panel, create_panel, prompt_panel, size_panel

from ..generator_process import Generator
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

                layout.prop(context.scene, "dream_textures_project_use_object_prompts")
                layout.prop(context.scene, "dream_textures_project_framebuffer_arguments")
                if context.scene.dream_textures_project_framebuffer_arguments == 'color':
                    layout.prop(get_prompt(context), "strength")

                row = layout.row()
                row.scale_y = 1.5
                row.operator(ProjectDreamTexture.bl_idname, icon="MOD_UVPROJECT")
        return ActionsPanel
    yield create_panel('VIEW_3D', 'UI', DREAM_PT_dream_panel_projection.bl_idname, actions_panel, get_prompt)

def draw(context, init_img_path, segmentation_map, segmentation_prompts, image_texture_node, material, cleanup):
    """
    Access the depth information from the current viewport, and pass it to `Generator.depth_to_image`.
    """
    try:
        # Read the depth information
        framebuffer = gpu.state.active_framebuffer_get()
        viewport = gpu.state.viewport_get()
        width, height = viewport[2], viewport[3]
        depth = np.asarray(framebuffer.read_depth(0, 0, width, height).to_list())

        depth = 1 - np.interp(depth, [depth.min(), depth.max()], [0, 1])

        # Downsample
        scaled_width = 512 if width < height else (512 * (width // height))
        scaled_height = 512 if height < width else (512 * (height // width))
        factor = max(width // scaled_width, height // scaled_height)

        depth = depth[::factor, ::factor]
        if segmentation_map is not None:
            segmentation_map = segmentation_map[::factor, ::factor]

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
            image_texture_node.image = texture
        
        future = Generator.shared().depth_to_image(
            depth=depth,
            image=init_img_path,
            segmentation_map=segmentation_map,
            segmentation_prompts=segmentation_prompts,
            **context.scene.dream_textures_project_prompt.generate_args()
        )
        future.add_response_callback(on_response)
        future.add_done_callback(on_done)

    finally:
        # The handle should always be removed.
        cleanup()

def draw_prompt_segmentation_map(width, height, scene, region_data):
    """
    Generate an image where each object is a different shade of gray based on their index.
    
    Index `0` would be colored `(0, 0, 0)`, index `12` colored `(12, 12, 12)`, and so on up to `255`.
    """
    offscreen = gpu.types.GPUOffScreen(width, height)

    with offscreen.bind():
        fb = gpu.state.active_framebuffer_get()
        fb.clear(color=(0.0, 0.0, 0.0, 0.0))
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(True)
        with gpu.matrix.push_pop():
            gpu.matrix.load_matrix(region_data.view_matrix)
            gpu.matrix.load_projection_matrix(region_data.window_matrix)

            for i, ob in enumerate(scene.objects):
                if (mesh := ob.data) is None:
                    continue
                
                mesh = mesh.copy()

                mesh.transform(ob.matrix_world)
                mesh.calc_loop_triangles()

                co = np.empty((len(mesh.vertices), 3), 'f')
                vertices = np.empty((len(mesh.loop_triangles), 3), 'i')

                mesh.vertices.foreach_get("co", co.ravel())
                mesh.loop_triangles.foreach_get("vertices", vertices.ravel())

                shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
                batch = batch_for_shader(
                    shader, 'TRIS',
                    {"pos": co},
                    indices=vertices,
                )
                shader.bind()
                shader.uniform_float("color", (i / 255., i / 255., i / 255., 1))
                batch.draw(shader)
        buffer = fb.read_color(0, 0, width, height, 4, 0, 'UBYTE')
    offscreen.free()
    return np.array([c for row in buffer for pix in row for c in pix]).reshape((height, width, 4))[:, :, 0]

class ProjectDreamTexture(bpy.types.Operator):
    bl_idname = "shade.dream_texture_project"
    bl_label = "Project Dream Texture"
    bl_description = "Automatically texture all selected objects using the depth buffer and Stable Diffusion"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return Generator.shared().can_use()

    def execute(self, context):
        # Use a segmentation map to specify which prompts go with which objects.
        if context.scene.dream_textures_project_use_object_prompts:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    for region in area.regions:
                        if region.type == 'WINDOW':
                            region_width, region_height = region.width, region.height
            segmentation_map = draw_prompt_segmentation_map(region_width, region_height, context.scene, context.region_data)
            segmentation_prompts = {
                i: ob.dream_textures_prompt.prompt for i, ob in enumerate(context.scene.objects)
            }
        else:
            segmentation_map = None
            segmentation_prompts = None
        # Render the viewport color if we're using img2img *and* depth2image.
        if context.scene.dream_textures_project_framebuffer_arguments == 'color':
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

        # Setup UVs from the current viewport angle.
        bpy.ops.uv.project_from_view()
        
        # Create a material and image texture nodes, and bind it to each selected object.
        material = bpy.data.materials.new(name="diffused-material")
        material.use_nodes = True
        image_texture_node = material.node_tree.nodes.new("ShaderNodeTexImage")
        material.node_tree.links.new(image_texture_node.outputs[0], material.node_tree.nodes['Principled BSDF'].inputs[0])
        for obj in bpy.context.selected_objects:
            if not hasattr(obj, "data") or not hasattr(obj.data, "materials"):
                continue
            if obj.data.materials:
                for slot in range(len(obj.data.materials)):
                    obj.data.materials[slot] = material
            else:
                obj.data.materials.append(material)

        # Use a draw handler to access the depth buffer.
        handle = None
        handle = bpy.types.SpaceView3D.draw_handler_add(
            draw,
            (
                context,
                init_img_path,
                segmentation_map,
                segmentation_prompts,
                image_texture_node,
                material,
                lambda: bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW'),
            ),
            'WINDOW',
            'POST_VIEW'
        )

        # Redraw the scene to ensure the texture changes propagate.
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
                return {'FINISHED'}

        return {'FINISHED'}
