import bpy
from bpy.props import CollectionProperty, StringProperty
from bpy_extras.io_utils import ImportHelper
import os
import webbrowser
import importlib.util
import site

from .absolute_path import absolute_path
from .operators.install_dependencies import InstallDependencies, UninstallDependencies
from .operators.open_latest_version import OpenLatestVersion
from .ui.presets import RestoreDefaultPresets, default_presets_missing
from .generator_process import Generator
from .generator_process.actions.prompt_to_image import Pipeline
from .generator_process.actions.huggingface_hub import DownloadStatus, ModelType
from .generator_process.actions.convert_original_stable_diffusion_to_diffusers import ModelConfig

is_downloading = False

class OpenURL(bpy.types.Operator):
    bl_idname = "dream_textures.open_url"
    bl_label = "Get Access Token"
    bl_description = ("Opens huggingface.co to the tokens page")
    bl_options = {"REGISTER", "INTERNAL"}

    url: bpy.props.StringProperty(name="URL")

    def execute(self, context):
        webbrowser.open(self.url)
        return {"FINISHED"}

_model_config_options = [(m.name, m.value, '') for m in ModelConfig]
class ImportWeights(bpy.types.Operator, ImportHelper):
    bl_idname = "dream_textures.import_weights"
    bl_label = "Import Checkpoint File"
    filename_ext = ".ckpt"
    filter_glob: bpy.props.StringProperty(
        default="*.ckpt",
        options={'HIDDEN'},
        maxlen=255,
    )
    model_config: bpy.props.EnumProperty(
        name="Model Config",
        items=_model_config_options
    )

    def execute(self, context):
        _, extension = os.path.splitext(self.filepath)
        if extension != '.ckpt':
            self.report({"ERROR"}, "Select a valid stable diffusion '.ckpt' file.")
            return {"FINISHED"}
        try:
            Generator.shared().convert_original_stable_diffusion_to_diffusers(self.filepath, ModelConfig[self.model_config]).result()
        except Exception as e:
            self.report({"ERROR"}, """Model conversion failed. Make sure you select the correct model configuration in the sidebar.
Press 'N' or click the gear icon in the top right of the file selection popup to reveal the sidebar.""")
            self.report({"ERROR"}, str(e))
        
        set_model_list('installed_models', Generator.shared().hf_list_installed_models().result())

        return {"FINISHED"}

class Model(bpy.types.PropertyGroup):
    bl_label = "Model"
    bl_idname = "dream_textures.Model"

    model: bpy.props.StringProperty(name="Model")
    model_base: bpy.props.StringProperty()
    downloads: bpy.props.IntProperty(name="Downloads")
    likes: bpy.props.IntProperty(name="Likes")
    model_type: bpy.props.EnumProperty(name="Model Type", items=[(t.name, t.name, '') for t in ModelType])

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
        if ModelType[item.model_type] != ModelType.UNKNOWN:
            split.label(text=item.model_type.replace('_', ' ').title())
        install_model = layout.operator(InstallModel.bl_idname, text="", icon="FILE_FOLDER" if is_installed else "IMPORT")
        install_model.model = item.model
        install_model.prefer_fp16_revision = data.prefer_fp16_revision

@staticmethod
def set_model_list(model_list: str, models: list):
    getattr(bpy.context.preferences.addons[__package__].preferences, model_list).clear()
    for model in models:
        m = getattr(bpy.context.preferences.addons[__package__].preferences, model_list).add()
        m.model = model.id
        m.model_base = os.path.basename(model.id)
        m.downloads = model.downloads
        m.likes = model.likes
        try:
            m.model_type = model.model_type.name
        except:
            pass

class ModelSearch(bpy.types.Operator):
    bl_idname = "dream_textures.model_search"
    bl_label = "Search"
    bl_description = ("Searches Hugging Face Hub for models")
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        
        return {"FINISHED"}

class InstallModel(bpy.types.Operator):
    bl_idname = "dream_textures.install_model"
    bl_label = "Install or Open"
    bl_description = ("Install or open a model from the cache")
    bl_options = {"REGISTER", "INTERNAL"}

    model: StringProperty(name="Model ID")
    prefer_fp16_revision: bpy.props.BoolProperty(name="", default=True)

    def execute(self, context):
        if os.path.exists(self.model):
            webbrowser.open(f"file://{self.model}")
        else:
            global is_downloading
            is_downloading = True
            f = Generator.shared().hf_snapshot_download(self.model, bpy.context.preferences.addons[__package__].preferences.hf_token, "fp16" if self.prefer_fp16_revision else None)
            def on_progress(_, response: DownloadStatus):
                bpy.context.preferences.addons[__package__].preferences.download_file = response.file
                bpy.context.preferences.addons[__package__].preferences.download_progress = int((response.index / response.total) * 100)
            def on_done(future):
                global is_downloading
                is_downloading = False
                set_model_list('installed_models', Generator.shared().hf_list_installed_models().result())
            def on_exception(_, exception):
                self.report({"ERROR"}, str(exception))
                raise exception
            f.add_response_callback(on_progress)
            f.add_done_callback(on_done)
            f.add_exception_callback(on_exception)
        return {"FINISHED"}

def _model_search(self, context):
    def on_done(future):
        set_model_list('model_results', future.result())
    Generator.shared().hf_list_models(self.model_query, self.hf_token).add_done_callback(on_done)

def _update_ui(self, context):
    if hasattr(context.area, "regions"):
        for region in context.area.regions:
            if region.type == "UI":
                region.tag_redraw()
    return None

def _template_model_download_progress(context, layout):
    global is_downloading
    preferences = context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences
    if is_downloading:
        progress_col = layout.column()
        progress_col.label(text=f"Downloading {preferences.download_file}")
        progress_col.prop(preferences, "download_progress", slider=True)
        progress_col.enabled = False
    return is_downloading

class StableDiffusionPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    dream_studio_key: StringProperty(name="DreamStudio Key")

    model_query: StringProperty(name="Search", update=_model_search)
    model_results: CollectionProperty(type=Model)
    active_model_result: bpy.props.IntProperty(name="Active Model", default=0)
    hf_token: StringProperty(name="HuggingFace Token")
    prefer_fp16_revision: bpy.props.BoolProperty(name="Prefer Half Precision Weights", description="Download fp16 weights if available for smaller file size. If you run with 'Half Precision' disabled, you should not use this setting", default=True)

    installed_models: CollectionProperty(type=Model)
    active_installed_model: bpy.props.IntProperty(name="Active Model", default=0)

    download_file: bpy.props.StringProperty(name="")
    download_progress: bpy.props.IntProperty(name="", min=0, max=100, subtype="PERCENTAGE", update=_update_ui)

    @staticmethod
    def register():
        if Pipeline.local_available():
            def on_done(future):
                set_model_list('installed_models', future.result())
            Generator.shared().hf_list_installed_models().add_done_callback(on_done)

    def draw(self, context):
        layout = self.layout

        weights_installed = len(self.installed_models) > 0

        if not weights_installed:
            layout.label(text="Complete the following steps to finish setting up the addon:")

        has_dependencies = len(os.listdir(absolute_path(".python_dependencies"))) > 2
        if has_dependencies:
            has_local = Pipeline.local_available()

            if has_local:
                if not _template_model_download_progress(context, layout):
                    conflicting_packages = ["wandb", "k_diffusion"]
                    conflicting_package_specs = {}
                    for package in conflicting_packages:
                        spec = importlib.util.find_spec(package)
                        if spec is not None:
                            conflicting_package_specs[package] = spec
                    if len(conflicting_package_specs) > 0:
                        conflicts_box = layout.box()
                        conflicts_box.label(text="WARNING", icon="ERROR")
                        conflicts_box.label(text=f"The following packages conflict with Dream Textures: {', '.join(conflicting_packages)}")
                        conflicts_box.label(text=f"You may need to run Blender as an administrator to remove these packages")
                        conflicts_box.operator(UninstallDependencies.bl_idname, text="Uninstall Conflicting Packages", icon="CANCEL").conflicts = ' '.join(conflicting_packages)
                        conflicts_box.label(text=f"If the button above fails, you can remove the following folders manually:")
                        for package in conflicting_packages:
                            if package not in conflicting_package_specs:
                                continue
                            location = conflicting_package_specs[package].submodule_search_locations[0]
                            conflicts_box.operator(OpenURL.bl_idname, text=f"Open '{location}'").url = f"file://{location}"

                    if not weights_installed:
                        default_weights_box = layout.box()
                        default_weights_box.label(text="You need to download at least one model.")
                        install_model = default_weights_box.operator(InstallModel.bl_idname, text="Download Stable Diffusion v2.1 (Recommended)", icon="IMPORT")
                        install_model.model = "stabilityai/stable-diffusion-2-1"
                        install_model.prefer_fp16_revision = self.prefer_fp16_revision

                    search_box = layout.box()
                    search_box.label(text="Find Models", icon="SETTINGS")
                    search_box.label(text="Search Hugging Face Hub for more compatible models.")

                    search_box.prop(self, "model_query", text="", icon="VIEWZOOM")
                    
                    if len(self.model_results) > 0:
                        search_box.template_list(PREFERENCES_UL_ModelList.__name__, "dream_textures_model_results", self, "model_results", self, "active_model_result")

                    search_box.label(text="Some models require authentication. Provide a token to download gated models.")

                    auth_row = search_box.row()
                    auth_row.prop(self, "hf_token", text="Token")
                    auth_row.operator(OpenURL.bl_idname, text="Get Your Token", icon="KEYINGSET").url = "https://huggingface.co/settings/tokens"
                    
                    search_box.prop(self, "prefer_fp16_revision")

                layout.template_list(PREFERENCES_UL_ModelList.__name__, "dream_textures_installed_models", self, "installed_models", self, "active_installed_model")
                layout.operator(ImportWeights.bl_idname, icon='IMPORT')
            
            dream_studio_box = layout.box()
            dream_studio_box.label(text=f"DreamStudio{' (Optional)' if has_local else ''}", icon="HIDE_OFF")
            dream_studio_box.label(text=f"Link to your DreamStudio account to run in the cloud{' instead of locally.' if has_local else '.'}")
            key_row = dream_studio_box.row()
            key_row.prop(self, "dream_studio_key", text="Key")
            key_row.operator(OpenURL.bl_idname, text="Find Your Key", icon="KEYINGSET").url = "https://beta.dreamstudio.ai/membership?tab=apiKeys"

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
        contributors_box.operator(OpenURL.bl_idname, text="See All Contributors", icon="URL").url = "https://github.com/carson-katri/dream-textures/graphs/contributors"

        if context.preferences.view.show_developer_ui: # If 'Developer Extras' is enabled, show addon development tools
            developer_box = layout.box()
            developer_box.label(text="Development Tools", icon="CONSOLE")
            warn_box = developer_box.box()
            warn_box.label(text="WARNING", icon="ERROR")
            warn_box.label(text="This section is for addon development only.")
            warn_box.label(text="Do not use any operators in this section unless you are setting up a development environment.")
            if has_dependencies:
                warn_box = developer_box.box()
                warn_box.label(text="Dependencies already installed. Only install below if you developing the addon", icon="CHECKMARK")
            developer_box.prop(context.scene, 'dream_textures_requirements_path')
            developer_box.operator_context = 'INVOKE_DEFAULT'
            developer_box.operator(InstallDependencies.bl_idname, icon="CONSOLE")
