import bpy
import gpu
import bmesh
import mathutils
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
import numpy as np

from .view_history import ImportPromptFile
from ..property_groups.dream_prompt import pipeline_options
from ..property_groups.project_perspective import ProjectPerspective
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

class ProjectDreamTexture(bpy.types.Operator):
    bl_idname = "shade.dream_texture_project"
    bl_label = "Project Dream Texture"
    bl_description = "Automatically texture all selected objects using the depth buffer and Stable Diffusion"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return Generator.shared().can_use()

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

        depth_maps = []
        for perspective in context.scene.dream_textures_project_perspectives:
            depth_maps.append(draw_depth_map(
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
            ))
        Generator.shared().depth_to_image(
            depth=depth_maps,
            image=[],
            **context.scene.dream_textures_project_prompt.generate_args()
        )

        return {'FINISHED'}
