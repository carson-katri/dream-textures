import bpy
from .dream_texture import create_panel, prompt_panel, advanced_panel
from ...property_groups.dream_prompt import backend_options
from ...generator_process.models import ModelType
from ...preferences import StableDiffusionPreferences

class RenderPropertiesPanel(bpy.types.Panel):
    """Panel for Dream Textures render properties"""
    bl_label = "Dream Textures"
    bl_idname = "DREAM_PT_dream_render_properties_panel"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(self, context):
        return context.scene.render.engine == 'CYCLES'

    def draw_header(self, context):
        self.layout.prop(context.scene, "dream_textures_render_properties_enabled", text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.active = context.scene.dream_textures_render_properties_enabled

        if len(backend_options(self, context)) > 1:
            layout.prop(context.scene.dream_textures_render_properties_prompt, "backend")
        layout.prop(context.scene.dream_textures_render_properties_prompt, 'model')
        layout.prop(context.scene.dream_textures_render_properties_prompt, "strength")
        layout.prop(context.scene, "dream_textures_render_properties_pass_inputs")
        if context.scene.dream_textures_render_properties_pass_inputs != 'color':
            if not bpy.context.view_layer.use_pass_z:
                box = layout.box()
                box.label(text="Z Pass Disabled", icon="ERROR")
                box.label(text="Enable the Z pass to use depth pass inputs")
                box.use_property_split = False
                box.prop(context.view_layer, "use_pass_z")
            
            models = list(filter(
                lambda m: m.model_base == context.scene.dream_textures_render_properties_prompt.model,
                context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences.installed_models
            ))
            if len(models) > 0 and ModelType[models[0].model_type] != ModelType.DEPTH:
                box = layout.box()
                box.label(text="Unsupported model", icon="ERROR")
                box.label(text="Select a depth model, such as 'stabilityai/stable-diffusion-2-depth'")

def render_properties_panels():
    yield RenderPropertiesPanel
    def get_prompt(context):
        return context.scene.dream_textures_render_properties_prompt
    space_type = RenderPropertiesPanel.bl_space_type
    region_type = RenderPropertiesPanel.bl_region_type
    panels = [
        *create_panel(space_type, region_type, RenderPropertiesPanel.bl_idname, prompt_panel, get_prompt, True),
        *create_panel(space_type, region_type, RenderPropertiesPanel.bl_idname, advanced_panel, get_prompt, True),
    ]
    for panel in panels:
        def draw_decorator(original):
            def draw(self, context):
                self.layout.enabled = context.scene.dream_textures_render_properties_enabled
                return original(self, context)
            return draw
        panel.draw = draw_decorator(panel.draw)
        if hasattr(panel, 'draw_header_preset'):
            panel.draw_header_preset = draw_decorator(panel.draw_header_preset)
        if hasattr(panel, 'draw_header'):
            panel.draw_header = draw_decorator(panel.draw_header)
        yield panel