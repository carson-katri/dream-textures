import bpy

class RenderPropertiesPanel(bpy.types.Panel):
    """Panel for Dream Textures render properties"""
    bl_label = "Dream Textures"
    bl_idname = "DREAM_PT_dream_render_properties_panel"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        self.layout.prop(context.scene, "dream_textures_render_properties_enabled", text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.active = context.scene.dream_textures_render_properties_enabled
        layout.prop(context.scene, "dream_textures_render_properties_prompt")