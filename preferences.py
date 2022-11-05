import bpy
from bpy.props import CollectionProperty, StringProperty
from bpy_extras.io_utils import ImportHelper
import os
import webbrowser
import shutil

from .absolute_path import CLIPSEG_WEIGHTS_PATH, REAL_ESRGAN_WEIGHTS_PATH, VAE_WEIGHTS_PATH, WEIGHTS_PATH, absolute_path, INPAINTING_WEIGHTS_PATH
from .operators.install_dependencies import InstallDependencies
from .operators.open_latest_version import OpenLatestVersion
from .property_groups.dream_prompt import DreamPrompt
from .ui.presets import RestoreDefaultPresets, default_presets_missing

model_types = [
    (WEIGHTS_PATH, 'Stable Diffusion', 'Normal Stable Diffusion weights file', 1),
    (INPAINTING_WEIGHTS_PATH, 'Stable Diffusion Inpainting', 'Inpainting-specific weights file', 2),
    (VAE_WEIGHTS_PATH, 'VAE', 'Variational autencoder weights file', 3),
    (REAL_ESRGAN_WEIGHTS_PATH, 'Upscaling', 'Real-ESRGAN weights file', 4),
    (CLIPSEG_WEIGHTS_PATH, 'Prompt Mask', 'CLIP segmentation weights file', 5),
]
model_type_extensions = {
    WEIGHTS_PATH: 'ckpt',
    INPAINTING_WEIGHTS_PATH: 'ckpt',
    VAE_WEIGHTS_PATH: 'ckpt',
    REAL_ESRGAN_WEIGHTS_PATH: 'pth',
    CLIPSEG_WEIGHTS_PATH: 'pth',
}

class OpenHuggingFace(bpy.types.Operator):
    bl_idname = "dream_textures.open_hugging_face"
    bl_label = "Download Weights from Hugging Face"
    bl_description = ("Opens huggingface.co to the download page for the model weights.")
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        webbrowser.open("https://huggingface.co/CompVis/stable-diffusion-v-1-4-original")
        return {"FINISHED"}

class ImportWeights(bpy.types.Operator, ImportHelper):
    bl_idname = "dream_textures.import_weights"
    bl_label = "Import Model Weights"
    filename_ext = ".ckpt;.pth"
    filter_glob: bpy.props.StringProperty(
        default="*.ckpt;*.pth",
        options={'HIDDEN'},
        maxlen=255,
    )
    model_type: bpy.props.EnumProperty(name="Model Type", items=model_types, description="The type of model the checkpoint file is for")

    def execute(self, context):
        path = os.path.dirname(self.model_type)
        if not os.path.exists(path):
            os.mkdir(path)
        _, extension = os.path.splitext(self.filepath)
        if extension != f'.{model_type_extensions[self.model_type]}':
            self.report({"ERROR"}, f"Select a valid '.{model_type_extensions[self.model_type]}' file.")
            return {"FINISHED"}
        shutil.copy(self.filepath, self.model_type)
        
        return {"FINISHED"}

class DeleteSelectedWeights(bpy.types.Operator):
    bl_idname = "dream_textures.delete_selected_weights"
    bl_label = "Delete Selected Weights"
    bl_options = {'REGISTER', 'INTERNAL'}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        os.remove(context.preferences.addons[__package__].preferences.weights[context.preferences.addons[__package__].preferences.active_weights].path)
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

class WeightsFile(bpy.types.PropertyGroup):
    bl_label = "Weights File"
    bl_idname = "dream_textures.WeightsFile"

    path: bpy.props.StringProperty(name="Path")
    model_type: bpy.props.EnumProperty(name="Model Type", items=model_types)

class PREFERENCES_UL_WeightsFileList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        layout.label(text=os.path.split(item.path)[1])
        layout.label(text=next(filter(lambda x: x[0] == item.model_type, model_types))[1])

class StableDiffusionPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    history: CollectionProperty(type=DreamPrompt)
    
    weights: CollectionProperty(type=WeightsFile)
    active_weights: bpy.props.IntProperty(name="Active Weights", default=0)

    dream_studio_key: StringProperty(name="DreamStudio Key")

    def draw(self, context):
        layout = self.layout

        self.weights.clear()
        for model_type in model_types:
            for path in filter(lambda f: f.endswith('.ckpt') or f.endswith('.pth'), os.listdir(model_type[0])):
                weights_file = self.weights.add()
                weights_file.path = os.path.join(model_type[0], path)
                weights_file.model_type = model_type[0]
        weights_installed = len(os.listdir(WEIGHTS_PATH)) > 1

        if not weights_installed:
            layout.label(text="Complete the following steps to finish setting up the addon:")

        if len(os.listdir(absolute_path(".python_dependencies"))) < 2:
            missing_sd_box = layout.box()
            missing_sd_box.label(text="Dependencies Missing", icon="ERROR")
            missing_sd_box.label(text="You've likely downloaded source instead of release by accident.")
            missing_sd_box.label(text="Follow the instructions to install for your platform.")
            missing_sd_box.operator(OpenLatestVersion.bl_idname, text="Download Latest Release")
            return

        has_local = len(os.listdir(absolute_path("stable_diffusion"))) > 0
        if has_local:
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
            model_weights_box.operator(ImportWeights.bl_idname, text="Import Model Weights", icon="IMPORT")
            model_weights_box.template_list(PREFERENCES_UL_WeightsFileList.__name__, "dream_textures_weights", self, "weights", self, "active_weights")
            model_weights_box.operator(DeleteSelectedWeights.bl_idname, text="Delete Selected Weights", icon="X")
        
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