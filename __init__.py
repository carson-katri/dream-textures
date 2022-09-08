# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name": "Dream Textures",
    "author": "Carson Katri",
    "description": "Use Stable Diffusion to generate unique textures straight from the shader editor.",
    "warning": "Requires installation of dependencies",
    "blender": (2, 80, 0),
    "version": (0, 0, 1),
    "location": "",
    "warning": "",
    "category": "Node"
}

import bpy
import os
import sys
import sysconfig
import subprocess
import importlib
import requests
import tarfile
import webbrowser
import numpy as np

dependencies_installed = False

def import_module(module_name, global_name=None, reload=True):
    """
    Import a module.
    :param module_name: Module to import.
    :param global_name: (Optional) Name under which the module is imported. If None the module_name will be used.
       This allows to import under a different name with the same effect as e.g. "import numpy as np" where "np" is
       the global_name under which the module can be accessed.
    :raises: ImportError and ModuleNotFoundError
    """
    if global_name is None:
        global_name = module_name

    if global_name in globals():
        importlib.reload(globals()[global_name])
    else:
        # Attempt to import the module and assign it to globals dictionary. This allow to access the module under
        # the given name, just like the regular import would.
        globals()[global_name] = importlib.import_module(module_name)


def install_pip():
    """
    Installs pip if not already present. Please note that ensurepip.bootstrap() also calls pip, which adds the
    environment variable PIP_REQ_TRACKER. After ensurepip.bootstrap() finishes execution, the directory doesn't exist
    anymore. However, when subprocess is used to call pip, in order to install a package, the environment variables
    still contain PIP_REQ_TRACKER with the now nonexistent path. This is a problem since pip checks if PIP_REQ_TRACKER
    is set and if it is, attempts to use it as temp directory. This would result in an error because the
    directory can't be found. Therefore, PIP_REQ_TRACKER needs to be removed from environment variables.
    :return:
    """

    try:
        # Check if pip is already installed
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True)
    except subprocess.CalledProcessError:
        import ensurepip

        ensurepip.bootstrap()
        os.environ.pop("PIP_REQ_TRACKER", None)


def install_and_import_module(module_name, package_name=None, global_name=None):
    """
    Installs the package through pip and attempts to import the installed module.
    :param module_name: Module to import.
    :param package_name: (Optional) Name of the package that needs to be installed. If None it is assumed to be equal
       to the module_name.
    :param global_name: (Optional) Name under which the module is imported. If None the module_name will be used.
       This allows to import under a different name with the same effect as e.g. "import numpy as np" where "np" is
       the global_name under which the module can be accessed.
    :raises: subprocess.CalledProcessError and ImportError
    """
    if package_name is None:
        package_name = module_name

    if global_name is None:
        global_name = module_name

    # Blender disables the loading of user site-packages by default. However, pip will still check them to determine
    # if a dependency is already installed. This can cause problems if the packages is installed in the user
    # site-packages and pip deems the requirement satisfied, but Blender cannot import the package from the user
    # site-packages. Hence, the environment variable PYTHONNOUSERSITE is set to disallow pip from checking the user
    # site-packages. If the package is not already installed for Blender's Python interpreter, it will then try to.
    # The paths used by pip can be checked with `subprocess.run([bpy.app.binary_path_python, "-m", "site"], check=True)`

    # Create a copy of the environment variables and modify them for the subprocess call
    environ_copy = dict(os.environ)
    environ_copy["PYTHONNOUSERSITE"] = "1"

    subprocess.run([sys.executable, "-m", "pip", "install", package_name], check=True, env=environ_copy)

    # The installation succeeded, attempt to import the module again
    import_module(module_name, global_name)

def install_and_import_requirements():
    """
    Installs all modules in the 'requirements.txt' file.
    """
    environ_copy = dict(os.environ)
    environ_copy["PYTHONNOUSERSITE"] = "1"

    response = requests.get(f"https://www.python.org/ftp/python/{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}/Python-{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}.tgz")
    open('python-devel.tgz', 'wb').write(response.content)
    python_devel_tgz = tarfile.open('python-devel.tgz')
    python_include_dir = sysconfig.get_paths()['include']
    def members(tf):
        prefix = f"Python-{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}/Include/"
        l = len(prefix)
        for member in tf.getmembers():
            if member.path.startswith(prefix):
                member.path = member.path[l:]
                yield member
    python_devel_tgz.extractall(path=python_include_dir, members=members(python_devel_tgz))

    subprocess.run([sys.executable, "-m", "pip", "install", "-r", os.path.abspath("stable_diffusion/requirements.txt")], check=True, env=environ_copy, cwd=os.path.abspath("stable_diffusion/"))

class StableDiffusionInstallDependencies(bpy.types.Operator):
    bl_idname = "stable_diffusion.install_dependencies"
    bl_label = "Install dependencies"
    bl_description = ("Downloads and installs the required python packages for this add-on. "
                      "Internet connection is required. Blender may have to be started with "
                      "elevated permissions in order to install the package")
    bl_options = {"REGISTER", "INTERNAL"}

    @classmethod
    def poll(self, context):
        return not dependencies_installed

    def execute(self, context):
        try:
            install_pip()
            install_and_import_requirements()
        except (subprocess.CalledProcessError, ImportError) as err:
            self.report({"ERROR"}, str(err))
            return {"CANCELLED"}

        global dependencies_installed
        dependencies_installed = True

        for cls in classes:
            bpy.utils.register_class(cls)
        bpy.types.NODE_HT_header.remove(shader_menu_draw)

        return {"FINISHED"}

class OpenHuggingFace(bpy.types.Operator):
    bl_idname = "stable_diffusion.open_hugging_face"
    bl_label = "Download Weights from Hugging Face"
    bl_description = ("Opens huggingface.co to the download page for the model weights.")
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        webbrowser.open("https://huggingface.co/CompVis/stable-diffusion-v-1-4-original")
        
        return {"FINISHED"}

weights_path = os.path.abspath("stable_diffusion/models/ldm/stable-diffusion-v1/model.ckpt")

class OpenWeightsDirectory(bpy.types.Operator):
    bl_idname = "stable_diffusion.open_weights_directory"
    bl_label = "Open Target Directory"
    bl_description = ("Opens the directory that should contain the 'model.ckpt' file.")
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        path = os.path.dirname(weights_path)
        if not os.path.exists(path):
            os.mkdir(path)
        webbrowser.open(f"file:///{os.path.realpath(path)}")
        
        return {"FINISHED"}

class StableDiffusionPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    def draw(self, context):
        layout = self.layout

        weights_installed = os.path.exists(weights_path)
        
        if dependencies_installed and weights_installed:
            layout.label(text="Addon setup complete.")
        else:
            layout.label(text="Complete the following steps to finish setting up the addon:")

        dependencies_box = layout.box()
        dependencies_box.label(text="Install Dependencies", icon="IMPORT")
        if dependencies_installed:
            dependencies_box.label(text="Dependencies installed successfully.")
        else:
            dependencies_box.operator(StableDiffusionInstallDependencies.bl_idname, icon="CONSOLE")
            
        model_weights_box = layout.box()
        model_weights_box.label(text="Setup Model Weights", icon="SETTINGS")
        if weights_installed:
            model_weights_box.label(text="Model weights setup successfully.")
        else:
            model_weights_box.label(text="The model weights are not distributed with the addon.")
            model_weights_box.label(text="Follow the steps below to download and install them.")
            model_weights_box.label(text="1. Download the file 'sd-v1-4.ckpt'")
            model_weights_box.operator(OpenHuggingFace.bl_idname, icon="URL")
            model_weights_box.label(text="2. Place the checkpoint file in the addon folder with the name 'model.ckpt'")
            warning_box = model_weights_box.box()
            warning_box.label(text="Make sure the file is renamed to 'model.ckpt', not 'sd-v1-4.ckpt'", icon="ERROR")
            model_weights_box.operator(OpenWeightsDirectory.bl_idname, icon="FOLDER_REDIRECT")

        if dependencies_installed and weights_installed:
            is_valid_box = layout.box()
            is_valid_box.label(text="Validate Installation", icon="EXPERIMENTAL")
            if is_install_valid is not None:
                if is_install_valid:
                    is_valid_box.label(text="Install validation succeeded.", icon="CHECKMARK")
                else:
                    is_valid_box.label(text="Install validation failed.", icon="ERROR")
            else:
                is_valid_box.operator(ValidateInstallation.bl_idname)

is_install_valid = None

class ValidateInstallation(bpy.types.Operator):
    bl_idname = "stable_diffusion.test_generation"
    bl_label = "Validate Installation"
    bl_description = ("Tests importing the generator to locate any errors with the installation.")
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        global is_install_valid
        try:
            from .stable_diffusion.ldm.simplet2i import T2I
            
            is_install_valid = True
        except:
            is_install_valid = False

        return {"FINISHED"}

preference_classes = (StableDiffusionInstallDependencies,
                      OpenHuggingFace,
                      OpenWeightsDirectory,
                      ValidateInstallation,
                      StableDiffusionPreferences)

from bpy.props import StringProperty, FloatProperty, IntProperty, EnumProperty, BoolProperty, PointerProperty

sampler_options = [
    ("ddim", "DDIM", "", 1),
    ("plms", "PLMS", "", 2),
    ("k_lms", "KLMS", "", 3),
    ("k_dpm_2", "KDPM_2", "", 4),
    ("k_dpm_2_a", "KDPM_2A", "", 5),
    ("k_euler", "KEULER", "", 6),
    ("k_euler_a", "KEULER_A", "", 7),
    ("k_heun", "KHEUN", "", 8),
]

from omegaconf import OmegaConf

class DreamTexture(bpy.types.Operator):
    bl_idname = "shade.dream_texture"
    bl_label = "Dream Texture"
    bl_description = "Generate a texture with AI"
    bl_options = {'REGISTER', 'UNDO'}

    prompt: StringProperty(name="Prompt")
    iterations: IntProperty(name="Iterations", default=1, min=1)
    steps: IntProperty(name="Steps", default=25, min=1)
    cfgscale: FloatProperty(name="CFG Scale", default=7.5)
    sampler: EnumProperty(name="Sampler", items=sampler_options, default=3)
    width: IntProperty(name="Width", default=512)
    height: IntProperty(name="Height", default=512)
    seed: IntProperty(name="Seed", default=-1)
    
    # Init Image
    strength: FloatProperty(name="Strength", default=0.75, min=0, max=1)
    fit: BoolProperty(name="Fit to width/height", default=True)

    @classmethod
    def poll(self, context):
        return True
    
    def invoke(self, context, event):
        weights_installed = os.path.exists(weights_path)
        if not weights_installed or not dependencies_installed:
            self.report({'ERROR'}, "Please complete setup in the preferences window.")
            return {"FINISHED"}
        else:
            return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "prompt")
        
        size_box = layout.box()
        size_box.label(text="Size")
        size_box.prop(self, "width")
        size_box.prop(self, "height")
        
        init_img_box = layout.box()
        init_img_box.label(text="Init Image")
        init_img_box.template_ID(context.scene, "init_img", open="image.open")
        init_img_box.prop(self, "strength")
        init_img_box.prop(self, "fit")

        advanced_box = layout.box()
        advanced_box.label(text="Advanced")
        advanced_box.prop(self, "seed")
        advanced_box.prop(self, "iterations")
        advanced_box.prop(self, "steps")
        advanced_box.prop(self, "cfgscale")
        advanced_box.prop(self, "sampler")

    def cancel(self, context):
        pass

    def execute(self, context):
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        from .stable_diffusion.ldm.simplet2i import T2I
        
        config  = os.path.abspath('stable_diffusion/configs/models.yaml')
        model   = 'stable-diffusion-1.4'

        models  = OmegaConf.load(config)
        width   = models[model].width
        height  = models[model].height
        config  = 'stable_diffusion/' + models[model].config
        weights = 'stable_diffusion/' + models[model].weights

        t2i = T2I(
            width=width,
            height=height,
            sampler_name=self.sampler,
            weights=weights,
            full_precision=True,
            config=config,
            grid=False,
            
            latent_diffusion_weights=False,
            embedding_path=None,
            device_type='cuda'
        )

        t2i.load_model()

        def image_writer(image, seed, upscaled=False):
            def pil_to_image(pil_image, name='NewImage'):
                '''
                PIL image pixels is 2D array of byte tuple (when mode is 'RGB', 'RGBA') or byte (when mode is 'L')
                bpy image pixels is flat array of normalized values in RGBA order
                '''
                # setup PIL image conversion
                width = pil_image.width
                height = pil_image.height
                byte_to_normalized = 1.0 / 255.0
                # create new image
                bpy_image = bpy.data.images.new(name, width=width, height=height)

                # convert Image 'L' to 'RGBA', normalize then flatten 
                bpy_image.pixels[:] = (np.asarray(pil_image.convert('RGBA'),dtype=np.float32) * byte_to_normalized).ravel()
                bpy_image.pack()

                return bpy_image

            nodes = context.material.node_tree.nodes
            texture_node = nodes.new("ShaderNodeTexImage")
            texture_node.image = pil_to_image(image, name=f"{seed}")
            nodes.active = texture_node

        t2i.prompt2image(
           prompt=self.prompt,                         # prompt string (no default)
           iterations=self.iterations,                     # iterations (1); image count=iterations
           steps=self.steps,                          # refinement steps per iteration
           seed=None,                           # seed for random number generator
           width=self.width,                          # width of image, in multiples of 64 (512)
           height=self.height,                         # height of image, in multiples of 64 (512)
           cfg_scale=self.cfgscale,                      # how strongly the prompt influences the image (7.5) (must be >1)
           init_img=None,                       # path to an initial image - its dimensions override width and height
           strength=self.strength,                       # strength for noising/unnoising init_img. 0.0 preserves image exactly, 1.0 replaces it completely
           gfpgan_strength=0.5,                # strength for GFPGAN. 0.0 preserves image exactly, 1.0 replaces it completely
           ddim_eta=0.0,                       # image randomness (eta=0.0 means the same seed always produces the same image)
           step_callback=None,                  # a function or method that will be called each step
           image_callback=image_writer,                 # a function or method that will be called each time an image is generated
           with_variations=None,                # a weighted list [(seed_1, weight_1), (seed_2, weight_2), ...] of variations which should be applied before doing any generation
           variation_amount=0.0,               # optional 0-1 value to slerp from -S noise to random noise (allows variations on an image)
        )

        return {'FINISHED'}

# Menu
class ShaderMenu(bpy.types.Menu):
    bl_idname = "OBJECT_MT_dream_textures"
    bl_label = "Dream Textures"

    def draw(self, context):
        layout = self.layout

        layout.operator(DreamTexture.bl_idname)

def shader_menu_draw(self, context):
    self.layout.menu(ShaderMenu.bl_idname)

classes = (
    DreamTexture,
    ShaderMenu,
)

def register():
    sys.path.insert(0, os.path.abspath("stable_diffusion"))
    sys.path.insert(0, os.path.abspath("stable_diffusion/src/clip"))
    sys.path.insert(0, os.path.abspath("stable_diffusion/src/k-diffusion"))
    sys.path.insert(0, os.path.abspath("stable_diffusion/src/taming-transformers"))

    global dependencies_installed
    dependencies_installed = False

    bpy.types.Scene.init_img = PointerProperty(name="Init Image", type=bpy.types.Image)

    for cls in preference_classes:
        bpy.utils.register_class(cls)

    try:
        # Check if the last dependency is installed.
        importlib.import_module("transformers")
        dependencies_installed = True
    except ModuleNotFoundError:
        # Don't register other panels, operators etc.
        return

    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.NODE_HT_header.append(shader_menu_draw)

def unregister():
    for cls in preference_classes:
        bpy.utils.unregister_class(cls)

    if dependencies_installed:
        for cls in classes:
            bpy.utils.unregister_class(cls)
        bpy.types.NODE_HT_header.remove(shader_menu_draw)


if __name__ == "__main__":
    register()