import bpy
import os
import sys
import importlib
import sysconfig
import subprocess
import requests
import tarfile

from ..absolute_path import absolute_path

dependencies_installed = False

def are_dependencies_installed():
    global dependencies_installed
    return dependencies_installed

def set_dependencies_installed(are_installed):
    global dependencies_installed
    dependencies_installed = are_installed

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

def install_and_import_requirements(requirements_txt=None):
    """
    Installs all modules in the 'requirements.txt' file.
    """
    environ_copy = dict(os.environ)
    environ_copy["PYTHONNOUSERSITE"] = "1"

    python_devel_tgz_path = absolute_path('python-devel.tgz')
    response = requests.get(f"https://www.python.org/ftp/python/{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}/Python-{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}.tgz")
    open(python_devel_tgz_path, 'wb').write(response.content)
    python_devel_tgz = tarfile.open(python_devel_tgz_path)
    python_include_dir = sysconfig.get_paths()['include']
    def members(tf):
        prefix = f"Python-{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}/Include/"
        l = len(prefix)
        for member in tf.getmembers():
            if member.path.startswith(prefix):
                member.path = member.path[l:]
                yield member
    python_devel_tgz.extractall(path=python_include_dir, members=members(python_devel_tgz))

    requirements_path = requirements_txt
    if requirements_path is None:
        if sys.platform == 'darwin': # Use MPS dependencies list on macOS
            requirements_path = 'stable_diffusion/requirements-mac-MPS-CPU.txt'
        else: # Use CUDA dependencies by default on Linux/Windows.
            # These are not the submodule dependencies from the `development` branch, but use the `main` branch deps for PyTorch 1.11.0.
            requirements_path = 'requirements-win-torch-1-11-0.txt'
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", absolute_path(requirements_path), "--no-cache-dir"], check=True, env=environ_copy, cwd=absolute_path("stable_diffusion/"))

class InstallDependencies(bpy.types.Operator):
    bl_idname = "stable_diffusion.install_dependencies"
    bl_label = "Install dependencies"
    bl_description = ("Downloads and installs the required python packages for this add-on. "
                      "Internet connection is required. Blender may have to be started with "
                      "elevated permissions in order to install the package")
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        # Open the console so we can watch the progress.
        if sys.platform == 'win32':
            bpy.ops.wm.console_toggle()

        try:
            install_pip()
            install_and_import_requirements(requirements_txt=context.scene.dream_textures_requirements_path)
        except (subprocess.CalledProcessError, ImportError) as err:
            self.report({"ERROR"}, str(err))
            return {"CANCELLED"}

        global dependencies_installed
        dependencies_installed = True

        subprocess.run([sys.executable, absolute_path("stable_diffusion/scripts/preload_models.py")], check=True, cwd=absolute_path("stable_diffusion/"))

        return {"FINISHED"}