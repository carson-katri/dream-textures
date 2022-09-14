import bpy
from ..property_groups.dream_prompt import sampler_options
from ..preferences import StableDiffusionPreferences

class ViewHistory(bpy.types.Operator):
    bl_idname = "shade.dream_textures_history"
    bl_label = "History"
    bl_description = "View and reload previous prompts"
    bl_options = {'REGISTER'}

    selection: bpy.props.IntProperty(name="Selection")

    @classmethod
    def poll(self, context):
        return True
    
    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=500)
    
    def draw(self, context):
        # split = self.layout.split(factor=0.95)
        # header = split.row()
        header = self.layout.row()
        header.label(text="Subject")
        header.label(text="Size")
        header.label(text="Steps")
        header.label(text="Sampler")
        self.layout.template_list("SCENE_UL_HistoryList", "", context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences, "history", context.scene, "dream_textures_history_selection")
        self.layout.operator(RecallHistoryEntry.bl_idname)
    
    def execute(self, context):
        return {"FINISHED"}
    
class SCENE_UL_HistoryList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.label(text=item.get_prompt_subject(), translate=False, icon_value=icon)
            layout.label(text=f"{item.width}x{item.height}", translate=False)
            layout.label(text=f"{item.steps} steps", translate=False)
            layout.label(text=next(x for x in sampler_options if x[0] == item.sampler)[1], translate=False)
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon_value=icon)

class RecallHistoryEntry(bpy.types.Operator):
    bl_idname = "shade.dream_textures_recall_history"
    bl_label = "Recall Prompt"
    bl_description = "Open the Dream Textures dialog with the historical properties filled in"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(self, context):
        return context.scene.dream_textures_history_selection is not None
    
    def execute(self, context):
        selection = context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences.history[context.scene.dream_textures_history_selection]
        for prop in selection.__annotations__.keys():
            if hasattr(context.scene.dream_textures_prompt, prop):
                setattr(context.scene.dream_textures_prompt, prop, getattr(selection, prop))
        bpy.ops.shade.dream_texture('INVOKE_DEFAULT')

        return {"FINISHED"}