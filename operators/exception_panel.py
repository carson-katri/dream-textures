import bpy

class ExceptionPanel(bpy.types.Operator):
    bl_idname = "shade.dream_textures_exception"
    bl_label = "Error"
    bl_description = "Shows error information"
    bl_options = {'REGISTER', 'UNDO'}

    exception_level: bpy.props.StringProperty()
    exception: bpy.props.StringProperty()

    @classmethod
    def poll(self, context):
        return True
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=500)
    
    def draw(self, context):
        self.layout.box().label(text=self.exception, icon=self.exception_level)
    
    def execute(self, context):
        return {"FINISHED"}