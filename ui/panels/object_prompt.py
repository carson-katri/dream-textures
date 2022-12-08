import bpy

class ObjectPromptPanel(bpy.types.Panel):
    bl_label = "Dream Textures"

    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "object"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True

        ob = context.object

        layout.prop(ob.dream_textures_prompt, "prompt")