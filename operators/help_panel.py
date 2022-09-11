import bpy
from ..help_section import help_section

class HelpPanel(bpy.types.Operator):
    bl_idname = "shade.dream_textures_help"
    bl_label = "Troubleshoot"
    bl_description = "Find fixes to some common issues"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(self, context):
        return True
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=500)
    
    def draw(self, context):
        help_section(self.layout, context)
    
    def execute(self, context):
        return {"FINISHED"}