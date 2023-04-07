import bpy
import gpu
from bl_ui.properties_render import RenderButtonsPanel
from bl_ui.properties_output import RenderOutputButtonsPanel
from bl_ui.properties_view_layer import ViewLayerButtonsPanel
import numpy as np
from ..ui.panels.dream_texture import optimization_panels
from .node_tree import DreamTexturesNodeTree
from ..engine import node_executor
from .annotations import depth

class DreamTexturesRenderEngine(bpy.types.RenderEngine):
    """A custom Dream Textures render engine, that uses Stable Diffusion and scene data to render images, instead of as a pass on top of Cycles."""

    bl_idname = "DREAM_TEXTURES"
    bl_label = "Dream Textures"
    bl_use_preview = False
    # bl_use_gpu_context = True

    def __init__(self):
        pass

    def __del__(self):
        pass

    def render(self, depsgraph):
        scene = depsgraph.scene

        def prepare_result(result):
            if len(result.shape) == 2:
                return np.concatenate(
                    (
                        np.stack((result,)*3, axis=-1),
                        np.ones((*result.shape, 1))
                    ),
                    axis=-1
                )
            else:
                return result
        
        result = self.begin_result(0, 0, scene.render.resolution_x, scene.render.resolution_y)
        layer = result.layers[0].passes["Combined"]
        self.update_result(result)

        try:
            progress = 0
            def node_begin(node):
                self.update_stats("Node", node.name)
            def node_update(response):
                if isinstance(response, np.ndarray):
                    node_result = prepare_result(response)
                    layer.rect = node_result.reshape(-1, node_result.shape[-1])
                    self.update_result(result)
            def node_end(_):
                nonlocal progress
                progress += 1
                self.update_progress(progress / len(scene.dream_textures_render_engine.node_tree.nodes))
            group_outputs = node_executor.execute(scene.dream_textures_render_engine.node_tree, depsgraph, node_begin=node_begin, node_update=node_update, node_end=node_end, test_break=self.test_break)
            node_result = group_outputs[0][1]
            for k, v in group_outputs:
                if type(v) == int or type(v) == str or type(v) == float:
                    self.get_result().stamp_data_add_field(k, str(v))
            node_result = prepare_result(node_result)
        except Exception as error:
            self.report({'ERROR'}, str(error))
            raise error

        layer.rect = node_result.reshape(-1, node_result.shape[-1])

        if "Depth" in result.layers[0].passes:
            z = depth.render_depth_map(depsgraph, invert=True)
            result.layers[0].passes["Depth"].rect = z.reshape((scene.render.resolution_x * scene.render.resolution_y, 1))
        
        self.end_result(result)
    
    def update_render_passes(self, scene=None, renderlayer=None):
        self.register_pass(scene, renderlayer, "Combined", 4, "RGBA", 'COLOR')
        self.register_pass(scene, renderlayer, "Depth", 1, "Z", 'VALUE')

class NewEngineNodeTree(bpy.types.Operator):
    bl_idname = "dream_textures.new_engine_node_tree"
    bl_label = "New Node Tree"

    def execute(self, context):
        bpy.ops.node.new_node_tree(type="DreamTexturesNodeTree")
        return {'FINISHED'}

def draw_device(self, context):
    scene = context.scene
    layout = self.layout
    layout.use_property_split = True
    layout.use_property_decorate = False

    if context.engine == DreamTexturesRenderEngine.bl_idname:
        layout.template_ID(scene.dream_textures_render_engine, "node_tree", text="Node Tree", new=NewEngineNodeTree.bl_idname)

def _poll_node_tree(self, value):
    return value.bl_idname == "DreamTexturesNodeTree"
class DreamTexturesRenderEngineProperties(bpy.types.PropertyGroup):
    node_tree: bpy.props.PointerProperty(type=DreamTexturesNodeTree, name="Node Tree", poll=_poll_node_tree)

def engine_panels():
    bpy.types.RENDER_PT_output.COMPAT_ENGINES.add(DreamTexturesRenderEngine.bl_idname)
    bpy.types.RENDER_PT_color_management.COMPAT_ENGINES.add(DreamTexturesRenderEngine.bl_idname)
    bpy.types.RENDER_PT_stamp.COMPAT_ENGINES.add(DreamTexturesRenderEngine.bl_idname)
    bpy.types.RENDER_PT_format.COMPAT_ENGINES.add(DreamTexturesRenderEngine.bl_idname)
    bpy.types.DATA_PT_lens.COMPAT_ENGINES.add(DreamTexturesRenderEngine.bl_idname)
    def get_prompt(context):
        return context.scene.dream_textures_engine_prompt
    class RenderPanel(bpy.types.Panel, RenderButtonsPanel):
        COMPAT_ENGINES = {DreamTexturesRenderEngine.bl_idname}

        def draw(self, context):
            self.layout.use_property_decorate = True
    class OutputPanel(bpy.types.Panel, RenderOutputButtonsPanel):
        COMPAT_ENGINES = {DreamTexturesRenderEngine.bl_idname}

        def draw(self, context):
            self.layout.use_property_decorate = True
    
    class ViewLayerPanel(bpy.types.Panel, ViewLayerButtonsPanel):
        COMPAT_ENGINES = {DreamTexturesRenderEngine.bl_idname}

        def draw(self, context):
            pass

    # Render Properties
    yield from optimization_panels(RenderPanel, 'engine', get_prompt, "")

    class NodeTreeInputsPanel(RenderPanel):
        """Create a subpanel for format options"""
        bl_idname = f"DREAM_PT_dream_panel_node_tree_inputs_engine"
        bl_label = "Inputs"

        def draw(self, context):
            super().draw(context)
            layout = self.layout
            layout.use_property_split = True

            if context.scene.dream_textures_render_engine.node_tree is not None:
                for input in context.scene.dream_textures_render_engine.node_tree.inputs:
                    layout.prop(input, "default_value", text=input.name)
    yield NodeTreeInputsPanel

    # View Layer
    class ViewLayerPassesPanel(ViewLayerPanel):
        bl_idname = "DREAM_PT_dream_panel_view_layer_passes"
        bl_label = "Passes"

        def draw(self, context):
            layout = self.layout
            layout.use_property_split = True
            layout.use_property_decorate = False

            view_layer = context.view_layer

            col = layout.column()
            col.prop(view_layer, "use_pass_combined")
            col.prop(view_layer, "use_pass_z")
            col.prop(view_layer, "use_pass_normal")
    yield ViewLayerPassesPanel

    # Bone properties
    class OpenPoseArmaturePanel(bpy.types.Panel):
        bl_idname = "DREAM_PT_dream_textures_armature_openpose"
        bl_label = "OpenPose"
        bl_space_type = 'PROPERTIES'
        bl_region_type = 'WINDOW'
        bl_context = "data"

        @classmethod
        def poll(cls, context):
            return context.armature
        
        def draw_header(self, context):
            bone = context.bone or context.edit_bone
            if bone:
                self.layout.prop(bone.dream_textures_openpose, "enabled", text="")

        def draw(self, context):
            layout = self.layout

            armature = context.armature

            p = armature.dream_textures_openpose

            row = layout.row()
            row.prop(p, "EAR_L", toggle=True)
            row.prop(p, "EYE_L", toggle=True)
            row.prop(p, "EYE_R", toggle=True)
            row.prop(p, "EAR_R", toggle=True)
            layout.prop(p, "NOSE", toggle=True)
            row = layout.row()
            row.prop(p, "SHOULDER_L", toggle=True)
            row.prop(p, "CHEST", toggle=True)
            row.prop(p, "SHOULDER_R", toggle=True)
            row = layout.row()
            row.prop(p, "ELBOW_L", toggle=True)
            row.separator()
            row.prop(p, "HIP_L", toggle=True)
            row.prop(p, "HIP_R", toggle=True)
            row.separator()
            row.prop(p, "ELBOW_R", toggle=True)
            row = layout.row()
            row.prop(p, "HAND_L", toggle=True)
            row.separator()
            row.prop(p, "KNEE_L", toggle=True)
            row.prop(p, "KNEE_R", toggle=True)
            row.separator()
            row.prop(p, "HAND_R", toggle=True)
            row = layout.row()
            row.prop(p, "FOOT_L", toggle=True)
            row.prop(p, "FOOT_R", toggle=True)

    yield OpenPoseArmaturePanel
    class OpenPoseBonePanel(bpy.types.Panel):
        bl_idname = "DREAM_PT_dream_textures_bone_openpose"
        bl_label = "OpenPose"
        bl_space_type = 'PROPERTIES'
        bl_region_type = 'WINDOW'
        bl_context = "bone"

        @classmethod
        def poll(cls, context):
            return context.bone and context.scene.render.engine == 'DREAM_TEXTURES'
        
        def draw_header(self, context):
            bone = context.bone
            if bone:
                self.layout.prop(bone.dream_textures_openpose, "enabled", text="")

        def draw(self, context):
            layout = self.layout
            layout.use_property_split = True

            bone = context.bone

            layout.enabled = bone.dream_textures_openpose.enabled
            layout.prop(bone.dream_textures_openpose, "bone")
            layout.prop(bone.dream_textures_openpose, "side")

    yield OpenPoseBonePanel

    class ADE20KObjectPanel(bpy.types.Panel):
        bl_idname = "DREAM_PT_dream_textures_object_ade20k"
        bl_label = "ADE20K Segmentation"
        bl_space_type = 'PROPERTIES'
        bl_region_type = 'WINDOW'
        bl_context = "object"

        @classmethod
        def poll(cls, context):
            return context.object and context.scene.render.engine == 'DREAM_TEXTURES'
        
        def draw_header(self, context):
            object = context.object
            if object:
                self.layout.prop(object.dream_textures_ade20k, "enabled", text="")

        def draw(self, context):
            layout = self.layout
            layout.use_property_split = True

            object = context.object

            layout.enabled = object.dream_textures_ade20k.enabled
            r = layout.split(factor=0.9)
            r.prop(object.dream_textures_ade20k, "annotation")
            c = r.column()
            c.enabled = False
            c.prop(object.dream_textures_ade20k, "color")

    yield ADE20KObjectPanel