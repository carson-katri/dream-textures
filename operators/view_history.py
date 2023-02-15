import bpy
from bpy_extras.io_utils import ImportHelper, ExportHelper
import json
import os
from ..property_groups.dream_prompt import DreamPrompt, scheduler_options
from ..preferences import StableDiffusionPreferences
    
class SCENE_UL_HistoryList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.label(text=item.get_prompt_subject(), translate=False, icon_value=icon)
            layout.label(text=f"{item.seed}", translate=False)
            layout.label(text=f"{item.width}x{item.height}", translate=False)
            layout.label(text=f"{item.steps} steps", translate=False)
            layout.label(text=next(x for x in scheduler_options if x[0] == item.scheduler)[1], translate=False)
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon_value=icon)

class RecallHistoryEntry(bpy.types.Operator):
    bl_idname = "shade.dream_textures_history_recall"
    bl_label = "Recall Prompt"
    bl_description = "Open the Dream Textures dialog with the historical properties filled in"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(self, context):
        return context.scene.dream_textures_history_selection is not None
    
    def execute(self, context):
        selection = context.scene.dream_textures_history[context.scene.dream_textures_history_selection]
        for prop in selection.__annotations__.keys():
            if hasattr(context.scene.dream_textures_prompt, prop):
                setattr(context.scene.dream_textures_prompt, prop, getattr(selection, prop))
            # when the seed of the promt is found in the available image datablocks, use that one in the open image editor
            # note: when there is more than one image with the seed in it's name, do nothing. Same when no image with that seed is available.
            if prop == 'hash':
                hash_string = str(getattr(selection, prop))
                existing_image = None
                # accessing custom properties for image datablocks in Blender is still a bit cumbersome
                for i in bpy.data.images:
                    if i.get('dream_textures_hash', None) == hash_string:
                        existing_image = i
                        break
                if existing_image is not None:
                    for area in context.screen.areas:
                        if area.type != 'IMAGE_EDITOR':
                            continue
                        area.spaces.active.image = existing_image

        return {"FINISHED"}

class ClearHistory(bpy.types.Operator):
    bl_idname = "shade.dream_textures_history_clear"
    bl_label = "Clear History"
    bl_description = "Removes all history entries"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        context.scene.dream_textures_history.clear()

        return {"FINISHED"}

class RemoveHistorySelection(bpy.types.Operator):
    bl_idname = "shade.dream_textures_history_remove_selection"
    bl_label = "Remove History Selection"
    bl_description = "Removes the selected history entry"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(self, context):
        return context.scene.dream_textures_history_selection is not None
    
    def execute(self, context):
        context.scene.dream_textures_history.remove(context.scene.dream_textures_history_selection)

        return {"FINISHED"}

class ExportHistorySelection(bpy.types.Operator, ExportHelper):
    bl_idname = "shade.dream_textures_history_export"
    bl_label = "Export Prompt"
    bl_description = "Exports the selected history entry to a JSON file"
    
    filename_ext = ".json"

    filter_glob: bpy.props.StringProperty(
        default="*.json",
        options={'HIDDEN'},
        maxlen=255,
    )

    @classmethod
    def poll(self, context):
        return context.scene.dream_textures_history_selection is not None
    
    def invoke(self, context, event):
        selection = context.scene.dream_textures_history[context.scene.dream_textures_history_selection]
        self.filepath = "untitled" if selection is None else selection.get_prompt_subject()
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        selection = context.scene.dream_textures_history[context.scene.dream_textures_history_selection]
        if selection is None:
            self.report({"ERROR"}, "No valid selection to export.")
            return {"FINISHED"}
        with open(self.filepath, 'w', encoding='utf-8') as target:
            args = {key: getattr(selection, key) for key in DreamPrompt.__annotations__}
            args["outpaint_origin"] = list(args["outpaint_origin"])
            json.dump(args, target, indent=4)

        return {"FINISHED"}

class ImportPromptFile(bpy.types.Operator, ImportHelper):
    bl_idname = "shade.dream_textures_import_prompt"
    bl_label = "Import Prompt"
    bl_description = "Imports a JSON file as a prompt"
    
    filename_ext = ".json"

    filter_glob: bpy.props.StringProperty(
        default="*.json",
        options={'HIDDEN'},
        maxlen=255,
    )
    
    def execute(self, context):
        _, extension = os.path.splitext(self.filepath)
        if extension != ".json":
            self.report({"ERROR"}, "Invalid prompt JSON file selected.")
            return {"FINISHED"}
        
        with open(self.filepath, 'r', encoding='utf-8') as target:
            args = json.load(target)
            for key, value in args.items():
                if hasattr(context.scene.dream_textures_prompt, key) and value is not None:
                    setattr(context.scene.dream_textures_prompt, key, value)

        return {"FINISHED"}