import bpy
from bpy.props import CollectionProperty
from bpy_extras.io_utils import ImportHelper
import os
import webbrowser
import shutil

from .absolute_path import WEIGHTS_PATH, absolute_path
from .operators.install_dependencies import InstallDependencies
from .operators.open_latest_version import OpenLatestVersion
from .property_groups.dream_prompt import DreamPrompt
from .ui.presets import RestoreDefaultPresets, default_presets_missing

class OpenHuggingFace(bpy.types.Operator):
    bl_idname = "dream_textures.open_hugging_face"
    bl_label = "Download Weights from Hugging Face"
    bl_description = ("Opens huggingface.co to the download page for the model weights.")
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        webbrowser.open("https://huggingface.co/CompVis/stable-diffusion-v-1-4-original")
        return {"FINISHED"}

class OpenWeightsDirectory(bpy.types.Operator, ImportHelper):
    bl_idname = "dream_textures.open_weights_directory"
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

class OpenContributors(bpy.types.Operator):
    bl_idname = "dream_textures.open_contributors"
    bl_label = "See All Contributors"

    def execute(self, context):
        webbrowser.open("https://github.com/carson-katri/dream-textures/graphs/contributors")
        return {"FINISHED"}

class StableDiffusionPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    history: CollectionProperty(type=DreamPrompt)

    def draw(self, context):
        layout = self.layout

        weights_installed = os.path.exists(WEIGHTS_PATH)

        if not weights_installed:
            layout.label(text="Complete the following steps to finish setting up the addon:")

        if not os.path.exists(absolute_path("stable_diffusion")) or len(os.listdir(absolute_path("stable_diffusion"))) < 5:
            missing_sd_box = layout.box()
            missing_sd_box.label(text="Stable diffusion is missing.", icon="ERROR")
            missing_sd_box.label(text="You've likely downloaded source instead of release by accident.")
            missing_sd_box.operator(OpenLatestVersion.bl_idname, text="Download Latest Release")
            return
        else:
            dependencies_box = layout.box()
            dependencies_box.label(text="Dependencies Located", icon="CHECKMARK")
            dependencies_box.label(text="All dependencies (except for model weights) are included in the release.")

        model_weights_box = layout.box()
        model_weights_box.label(text="Setup Model Weights", icon="SETTINGS")
        if weights_installed:
            model_weights_box.label(text="Model weights setup successfully.", icon="CHECKMARK")
        else:
            model_weights_box.label(text="The model weights are not distributed with the addon.")
            model_weights_box.label(text="Follow the steps below to download and install them.")
            model_weights_box.label(text="1. Download the file 'sd-v1-4.ckpt'")
            model_weights_box.operator(OpenHuggingFace.bl_idname, icon="URL")
            model_weights_box.label(text="2. Select the downloaded weights to install.")
            model_weights_box.operator(OpenWeightsDirectory.bl_idname, text="Import Model Weights", icon="IMPORT")

        if weights_installed:
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
        
        contributors_box = layout.box()
        contributors_box.label(text="Contributors", icon="COMMUNITY")
        contributors_box.label(text="Dream Textures is made possible by the contributors on GitHub.")
        contributors_box.operator(OpenContributors.bl_idname, icon="URL")

        if context.preferences.view.show_developer_ui: # If 'Developer Extras' is enabled, show addon development tools
            developer_box = layout.box()
            developer_box.label(text="Development Tools", icon="CONSOLE")
            developer_box.label(text="This section is for addon development only. You are seeing this because you have 'Developer Extras' enabled.")
            developer_box.label(text="Do not use any operators in this section unless you are setting up a development environment.")
            already_installed = len(os.listdir(absolute_path(".python_dependencies"))) > 0
            if already_installed:
                warn_box = developer_box.box()
                warn_box.label(text="Dependencies already installed. Only install below if you developing the addon", icon="CHECKMARK")
            developer_box.prop(context.scene, 'dream_textures_requirements_path')
            developer_box.operator(InstallDependencies.bl_idname, icon="CONSOLE")