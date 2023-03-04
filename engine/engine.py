import bpy
import gpu
from bl_ui.properties_render import RenderButtonsPanel
from bl_ui.properties_output import RenderOutputButtonsPanel
from ..ui.panels.dream_texture import create_panel, prompt_panel, advanced_panel, size_panel
from ..property_groups.dream_prompt import control_net_options
from .node_tree import DreamTexturesNodeTree

class DreamTexturesRenderEngine(bpy.types.RenderEngine):
    """A custom Dream Textures render engine, that uses Stable Diffusion and scene data to render images, instead of as a pass on top of Cycles."""

    bl_idname = "DREAM_TEXTURES"
    bl_label = "Dream Textures"
    bl_use_preview = False

    def __init__(self):
        pass

    def __del__(self):
        pass

    def render(self, depsgraph):
        scene = depsgraph.scene
        scale = scene.render.resolution_percentage / 100.0
        self.size_x = int(scene.render.resolution_x * scale)
        self.size_y = int(scene.render.resolution_y * scale)

        # Fill the render result with a flat color. The framebuffer is
        # defined as a list of pixels, each pixel itself being a list of
        # R,G,B,A values.
        if self.is_preview:
            color = [0.1, 0.2, 0.1, 1.0]
        else:
            color = [0.2, 0.1, 0.1, 1.0]

        pixel_count = self.size_x * self.size_y
        rect = [color] * pixel_count

        # Here we write the pixel values to the RenderResult
        result = self.begin_result(0, 0, self.size_x, self.size_y)
        layer = result.layers[0].passes["Combined"]
        layer.rect = rect
        self.end_result(result)
    
    def view_update(self, context, depsgraph):
        region = context.region
        view3d = context.space_data
        scene = depsgraph.scene

        # Get viewport dimensions
        dimensions = region.width, region.height

        if not self.scene_data:
            # First time initialization
            self.scene_data = []
            first_time = True

            # Loop over all datablocks used in the scene.
            for datablock in depsgraph.ids:
                pass
        else:
            first_time = False

            # Test which datablocks changed
            for update in depsgraph.updates:
                print("Datablock updated: ", update.id.name)

            # Test if any material was added, removed or changed.
            if depsgraph.id_type_updated('MATERIAL'):
                print("Materials updated")

        # Loop over all object instances in the scene.
        if first_time or depsgraph.id_type_updated('OBJECT'):
            for instance in depsgraph.object_instances:
                pass

    # For viewport renders, this method is called whenever Blender redraws
    # the 3D viewport. The renderer is expected to quickly draw the render
    # with OpenGL, and not perform other expensive work.
    # Blender will draw overlays for selection and editing on top of the
    # rendered image automatically.
    def view_draw(self, context, depsgraph):
        region = context.region
        scene = depsgraph.scene

        # Get viewport dimensions
        dimensions = region.width, region.height

        # Bind shader that converts from scene linear to display space,
        gpu.state.blend_set('ALPHA_PREMULT')
        self.bind_display_space_shader(scene)

        if not self.draw_data or self.draw_data.dimensions != dimensions:
            self.draw_data = CustomDrawData(dimensions)

        self.draw_data.draw()

        self.unbind_display_space_shader()
        gpu.state.blend_set('NONE')

def draw_device(self, context):
    scene = context.scene
    layout = self.layout
    layout.use_property_split = True
    layout.use_property_decorate = False

    if context.engine == DreamTexturesRenderEngine.bl_idname:
        layout.template_ID(scene.dream_textures_render_engine, "node_tree", text="Node Tree")

def _poll_node_tree(self, value):
    return value.bl_idname == "dream_textures.node_tree"
class DreamTexturesRenderEngineProperties(bpy.types.PropertyGroup):
    node_tree: bpy.props.PointerProperty(type=DreamTexturesNodeTree, name="Node Tree", poll=_poll_node_tree)

def engine_panels():
    bpy.types.RENDER_PT_output.COMPAT_ENGINES.add(DreamTexturesRenderEngine.bl_idname)
    def get_prompt(context):
        return context.scene.dream_textures_prompt
    class RenderPanel(bpy.types.Panel, RenderButtonsPanel):
        COMPAT_ENGINES = {DreamTexturesRenderEngine.bl_idname}
        def draw(self, context):
            self.layout.use_property_decorate = True
    class OutputPanel(bpy.types.Panel, RenderOutputButtonsPanel):
        COMPAT_ENGINES = {DreamTexturesRenderEngine.bl_idname}

        def draw(self, context):
            self.layout.use_property_decorate = True

    # Render Properties
    yield from advanced_panel(RenderPanel, 'engine', get_prompt)

    # Output Properties
    yield size_panel(OutputPanel, 'engine', get_prompt)