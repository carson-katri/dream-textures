import bpy
from ..property_groups.dream_prompt import sampler_options
from ..preferences import StableDiffusionPreferences
    
class SCENE_UL_HistoryList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            if item.prompt_structure_token_subject == "SCENE_UL_HistoryList_header":
                layout.label(text="Subject")
                layout.label(text="Size")
                layout.label(text="Steps")
                layout.label(text="Sampler")
            else:
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
        return context.scene.dream_textures_history_selection is not None and context.scene.dream_textures_history_selection > 0
    
    def execute(self, context):
        selection = context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences.history[context.scene.dream_textures_history_selection]
        for prop in selection.__annotations__.keys():
            if hasattr(context.scene.dream_textures_prompt, prop):
                setattr(context.scene.dream_textures_prompt, prop, getattr(selection, prop))

        return {"FINISHED"}