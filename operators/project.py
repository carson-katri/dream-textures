import bpy
import gpu
import numpy as np

from .view_history import ImportPromptFile
from ..property_groups.dream_prompt import backend_options
from ..generator_process.registrar import BackendTarget
from .open_latest_version import OpenLatestVersion, is_force_show_download, new_version_available

from ..ui.panels.dream_texture import advanced_panel, create_panel, prompt_panel, size_panel

from ..generator_process import Generator
from ..generator_process.actions.prompt_to_image import Pipeline

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

            if len(backend_options(self, context)) > 1:
                layout.prop(context.scene.dream_textures_project_prompt, "backend")
            if context.scene.dream_textures_project_prompt.backend == BackendTarget.LOCAL.name:
                layout.prop(context.scene.dream_textures_project_prompt, 'model')

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

                layout.prop(get_prompt(context), "strength")
                # layout.prop(context.scene, "dream_textures_projection_orbit_steps")

                row = layout.row()
                row.scale_y = 1.5
                row.operator(ProjectDreamTexture.bl_idname, icon="MOD_UVPROJECT")
        return ActionsPanel
    yield create_panel('VIEW_3D', 'UI', DREAM_PT_dream_panel_projection.bl_idname, actions_panel, get_prompt)

def draw(context, image_texture_node, cleanup):
    try:
        framebuffer = gpu.state.active_framebuffer_get()
        viewport = gpu.state.viewport_get()
        width, height = viewport[2], viewport[3]
        depth = np.asarray(framebuffer.read_depth(0, 0, width, height).to_list())

        depth = 1 - np.interp(depth, [depth.min(), depth.max()], [0, 1])

        scaled_width = 512 if width < height else (512 * (width // height))
        scaled_height = 512 if height < width else (512 * (height // width))
        factor = max(width // scaled_width, height // scaled_height)

        depth = depth[::factor, ::factor]

        texture = None

        def on_response(_, response):
            nonlocal texture
            if texture is not None:
                bpy.data.images.remove(texture)
            texture = bpy.data.images.new(name="diffused-image-texture", width=response.shape[1], height=response.shape[0])
            texture.pixels[:] = response.ravel()
            image_texture_node.image = texture

        def on_done(future):
            nonlocal texture
            if texture is not None:
                bpy.data.images.remove(texture)

            generated = future.result()
            if isinstance(generated, list):
                generated = generated[-1]
            texture = bpy.data.images.new(name="diffused-image-texture", width=generated.shape[1], height=generated.shape[0])
            texture.pixels[:] = generated.ravel()
            image_texture_node.image = texture
        
        future = Generator.shared().depth_to_image(
            Pipeline.STABLE_DIFFUSION,
            depth=depth,
            **context.scene.dream_textures_project_prompt.generate_args()
        )
        future.add_response_callback(on_response)
        future.add_done_callback(on_done)

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

    def execute(self, context):
        bpy.ops.uv.project_from_view()
        
        material = bpy.data.materials.new(name="diffused-material")
        material.use_nodes = True
        image_texture_node = material.node_tree.nodes.new("ShaderNodeTexImage")
        material.node_tree.links.new(image_texture_node.outputs[0], material.node_tree.nodes['Principled BSDF'].inputs[0])
        for obj in bpy.context.selected_objects:
            if not hasattr(obj, "data") or not hasattr(obj.data, "materials"):
                continue
            if obj.data.materials:
                print("Set mat slot")
                obj.data.materials[0] = material
            else:
                print("Append mat")
                obj.data.materials.append(material)

        handle = None
        handle = bpy.types.SpaceView3D.draw_handler_add(
            draw,
            (
                context,
                image_texture_node,
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
