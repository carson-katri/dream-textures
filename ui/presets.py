import bpy
from bpy.types import Panel, Operator, Menu
from bl_operators.presets import AddPresetBase
from bl_ui.utils import PresetPanel
import os
import shutil
from ..absolute_path import absolute_path

class DreamTexturesPresetPanel(PresetPanel, Panel):
    preset_operator = "script.execute_preset"

    # @staticmethod
    # def post_cb(context):
    #     # Modify an arbitrary built-in scene property to force a depsgraph
    #     # update, because add-on properties don't. (see T62325)
    #     render = context.scene.render
    #     render.filter_size = render.filter_size

class DREAM_PT_AdvancedPresets(DreamTexturesPresetPanel):
    bl_label = "Advanced Presets"
    preset_subdir = "dream_textures/advanced"
    preset_add_operator = "dream_textures.advanced_preset_add"

class DREAM_MT_AdvancedPresets(Menu):
    bl_label = 'Advanced Presets'
    preset_subdir = 'dream_textures/advanced'
    preset_operator = 'script.execute_preset'
    draw = Menu.draw_preset

class AddAdvancedPreset(AddPresetBase, Operator):
    bl_idname = 'dream_textures.advanced_preset_add'
    bl_label = 'Add Advanced Preset'
    preset_menu = 'DREAM_MT_AdvancedPresets'
    
    preset_subdir = 'dream_textures/advanced'
    
    preset_defines = ['prompt = bpy.context.scene.dream_textures_prompt']
    preset_values = [
        "prompt.precision",
        "prompt.random_seed",
        "prompt.seed",
        "prompt.steps",
        "prompt.cfg_scale",
        "prompt.sampler_name",
        "prompt.show_steps",
    ]

class RestoreDefaultPresets(Operator):
    bl_idname = "dream_textures.restore_default_presets"
    bl_label = "Restore Default Presets"
    bl_description = ("Restores all default presets provided by the addon.")
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        register_default_presets(force=True)
        return {"FINISHED"}

PRESETS_PATH = os.path.join(bpy.utils.user_resource('SCRIPTS'), 'presets/dream_textures/advanced')
DEFAULT_PRESETS_PATH = absolute_path('builtin_presets')
def register_default_presets(force=False):
    presets_path_exists = os.path.isdir(PRESETS_PATH)
    if not presets_path_exists or force:
        if not presets_path_exists:
            os.makedirs(PRESETS_PATH)
        for default_preset in os.listdir(DEFAULT_PRESETS_PATH):
            if not os.path.exists(os.path.join(PRESETS_PATH, default_preset)):
                shutil.copy(os.path.join(DEFAULT_PRESETS_PATH, default_preset), PRESETS_PATH)

def default_presets_missing():
    if not os.path.isdir(PRESETS_PATH):
        return True
    for default_preset in os.listdir(DEFAULT_PRESETS_PATH):
        if not os.path.exists(os.path.join(PRESETS_PATH, default_preset)):
            return True