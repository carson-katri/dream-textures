import bpy
import os
import sys

class NotifyResult(bpy.types.Operator):
    bl_idname = "shade.dream_textures_notify_result"
    bl_label = "Notify Result"
    bl_description = "Notifies of a generation completion or any error messages"
    bl_options = {'REGISTER'}

    exception: bpy.props.StringProperty(name="Exception", default="")

    def modal(self, context, event):
        if self.exception != "":
            self.report({'ERROR'}, f"""An error occurred while generating. Check the issues tab on GitHub to see if this has been reported before:

{self.exception}""")
            return {'CANCELLED'}
        else:
            return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        return {'FINISHED'}