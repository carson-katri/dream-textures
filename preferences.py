import bpy
from bpy.props import CollectionProperty, StringProperty
from bpy_extras.io_utils import ImportHelper
import os
import webbrowser
import shutil

from .absolute_path import WEIGHTS_PATH, absolute_path
from .operators.install_dependencies import InstallDependencies
from .operators.open_latest_version import OpenLatestVersion
from .property_groups.dream_prompt import DreamPrompt
from .ui.presets import RestoreDefaultPresets, default_presets_missing
from .generator_process import Generator
from .generator_process.registrar import BackendTarget

class OpenHuggingFace(bpy.types.Operator):
    bl_idname = "dream_textures.open_hugging_face"
    bl_label = "Get Access Token"
    bl_description = ("Opens huggingface.co to the tokens page")
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        webbrowser.open("https://huggingface.co/settings/tokens")
        return {"FINISHED"}

class ImportWeights(bpy.types.Operator, ImportHelper):
    bl_idname = "dream_textures.import_weights"
    bl_label = "Import Model Weights"
    filename_ext = ".ckpt"
    filter_glob: bpy.props.StringProperty(
        default="*.ckpt",
        options={'HIDDEN'},
        maxlen=255,
    )

    def execute(self, context):
        path = os.path.dirname(WEIGHTS_PATH)
        if not os.path.exists(path):
            os.mkdir(path)
        _, extension = os.path.splitext(self.filepath)
        if extension != '.ckpt':
            self.report({"ERROR"}, "Select a valid stable diffusion '.ckpt' file.")
            return {"FINISHED"}
        shutil.copy(self.filepath, WEIGHTS_PATH)
        
        return {"FINISHED"}

class DeleteSelectedWeights(bpy.types.Operator):
    bl_idname = "dream_textures.delete_selected_weights"
    bl_label = "Delete Selected Weights"
    bl_options = {'REGISTER', 'INTERNAL'}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        os.remove(os.path.join(WEIGHTS_PATH, context.preferences.addons[__package__].preferences.weights[context.preferences.addons[__package__].preferences.active_weights].name))
        return {"FINISHED"}

class OpenContributors(bpy.types.Operator):
    bl_idname = "dream_textures.open_contributors"
    bl_label = "See All Contributors"

    def execute(self, context):
        webbrowser.open("https://github.com/carson-katri/dream-textures/graphs/contributors")
        return {"FINISHED"}

class OpenDreamStudio(bpy.types.Operator):
    bl_idname = "dream_textures.open_dream_studio"
    bl_label = "Find Your Key"
    bl_description = ("Opens DreamStudio to the API key tab.")
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        webbrowser.open("https://beta.dreamstudio.ai/membership?tab=apiKeys")
        return {"FINISHED"}

class Model(bpy.types.PropertyGroup):
    bl_label = "Model"
    bl_idname = "dream_textures.Model"

    model: bpy.props.StringProperty(name="Model")
    downloads: bpy.props.IntProperty(name="Downloads")
    likes: bpy.props.IntProperty(name="Likes")

class PREFERENCES_UL_ModelList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        model_name = item.model
        is_installed = False
        if os.path.exists(item.model):
            model_name = os.path.basename(item.model).replace('models--', '').replace('--', '/')
            is_installed = True
        split = layout.split(factor=0.75)
        split.label(text=model_name)
        if item.downloads != -1:
            split.label(text=str(item.downloads), icon="IMPORT")
        if item.downloads != -1:
            split.label(text=str(item.likes), icon="HEART")
        layout.operator(InstallModel.bl_idname, text="", icon="FILE_FOLDER" if is_installed else "IMPORT").model = item.model

@staticmethod
def set_model_list(model_list: str, models: list):
    getattr(bpy.context.preferences.addons[__package__].preferences, model_list).clear()
    for model in models:
        m = getattr(bpy.context.preferences.addons[__package__].preferences, model_list).add()
        m.model = model.id
        m.downloads = model.downloads
        m.likes = model.likes

class ModelSearch(bpy.types.Operator):
    bl_idname = "dream_textures.model_search"
    bl_label = "Search"
    bl_description = ("Searches HuggingFace Hub for models")
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        
        return {"FINISHED"}

class InstallModel(bpy.types.Operator):
    bl_idname = "dream_textures.install_model"
    bl_label = "Install or Open"
    bl_description = ("Install or open a model from the cache")
    bl_options = {"REGISTER", "INTERNAL"}

    model: StringProperty(name="Model ID")

    def execute(self, context):
        if os.path.exists(self.model):
            webbrowser.open(f"file://{self.model}")
        else:
            _ = Generator.shared().hf_snapshot_download(self.model, bpy.context.preferences.addons[__package__].preferences.hf_token).result()
            set_model_list('installed_models', Generator.shared().hf_list_installed_models().result())
        return {"FINISHED"}

def _model_search(self, context):
    def on_done(future):
        set_model_list('model_results', future.result())
    Generator.shared().hf_list_models(self.model_query).add_done_callback(on_done)

class StableDiffusionPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    history: CollectionProperty(type=DreamPrompt)

    dream_studio_key: StringProperty(name="DreamStudio Key")

    model_query: StringProperty(name="Search", update=_model_search)
    model_results: CollectionProperty(type=Model)
    active_model_result: bpy.props.IntProperty(name="Active Model", default=0)
    hf_token: StringProperty(name="HuggingFace Token")

    installed_models: CollectionProperty(type=Model)
    active_installed_model: bpy.props.IntProperty(name="Active Model", default=0)

    @staticmethod
    def register():
        if BackendTarget.local_available():
            set_model_list('installed_models', Generator.shared().hf_list_installed_models().result())

    def draw(self, context):
        layout = self.layout

        weights_installed = len(self.installed_models) > 0

        if not weights_installed:
            layout.label(text="Complete the following steps to finish setting up the addon:")

        has_dependencies = len(os.listdir(absolute_path(".python_dependencies"))) > 2
        if has_dependencies:
            has_local = BackendTarget.local_available()

            if has_local:        
                search_box = layout.box()
                search_box.label(text="Find Models", icon="SETTINGS")
                search_box.label(text="Search HuggingFace Hub for compatible models.")
                
                auth_row = search_box.row()
                auth_row.prop(self, "hf_token", text="Token")
                auth_row.operator(OpenHuggingFace.bl_idname, text="Get Your Token", icon="KEYINGSET")

                search_box.prop(self, "model_query", text="", icon="VIEWZOOM")
                
                if len(self.model_results) > 0:
                    search_box.template_list(PREFERENCES_UL_ModelList.__name__, "dream_textures_model_results", self, "model_results", self, "active_model_result")

                layout.template_list(PREFERENCES_UL_ModelList.__name__, "dream_textures_installed_models", self, "installed_models", self, "active_installed_model")
            
            dream_studio_box = layout.box()
            dream_studio_box.label(text=f"DreamStudio{' (Optional)' if has_local else ''}", icon="HIDE_OFF")
            dream_studio_box.label(text=f"Link to your DreamStudio account to run in the cloud{' instead of locally.' if has_local else '.'}")
            key_row = dream_studio_box.row()
            key_row.prop(self, "dream_studio_key", text="Key")
            key_row.operator(OpenDreamStudio.bl_idname, text="Find Your Key", icon="KEYINGSET")

            if weights_installed or len(self.dream_studio_key) > 0:
                complete_box = layout.box()
                complete_box.label(text="Addon Setup Complete", icon="CHECKMARK")
                complete_box.label(text="To locate the interface:")
                complete_box.label(text="1. Open an Image Editor or Shader Editor space")
                complete_box.label(text="2. Enable 'View' > 'Sidebar'")
                complete_box.label(text="3. Select the 'Dream' tab")
            
            if default_presets_missing():
                presets_box = layout.box()
                presets_box.label(text="Default Presets", icon="PRESET")
                presets_box.label(text="It looks like you removed some of the default presets.")
                presets_box.label(text="You can restore them here.")
                presets_box.operator(RestoreDefaultPresets.bl_idname, icon="RECOVER_LAST")
        else:
            missing_dependencies_box = layout.box()
            missing_dependencies_box.label(text="Dependencies Missing", icon="ERROR")
            missing_dependencies_box.label(text="You've likely downloaded source instead of release by accident.")
            missing_dependencies_box.label(text="Follow the instructions to install for your platform.")
            missing_dependencies_box.operator(OpenLatestVersion.bl_idname, text="Download Latest Release")
        
        contributors_box = layout.box()
        contributors_box.label(text="Contributors", icon="COMMUNITY")
        contributors_box.label(text="Dream Textures is made possible by the contributors on GitHub.")
        contributors_box.operator(OpenContributors.bl_idname, icon="URL")

        if context.preferences.view.show_developer_ui: # If 'Developer Extras' is enabled, show addon development tools
            developer_box = layout.box()
            developer_box.label(text="Development Tools", icon="CONSOLE")
            developer_box.label(text="This section is for addon development only. You are seeing this because you have 'Developer Extras' enabled.")
            developer_box.label(text="Do not use any operators in this section unless you are setting up a development environment.")
            if has_dependencies:
                warn_box = developer_box.box()
                warn_box.label(text="Dependencies already installed. Only install below if you developing the addon", icon="CHECKMARK")
            developer_box.prop(context.scene, 'dream_textures_requirements_path')
            developer_box.operator(InstallDependencies.bl_idname, icon="CONSOLE")