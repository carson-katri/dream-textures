import bpy
import os
import sys
import importlib
import sysconfig
import subprocess
import requests
import tarfile

from ..absolute_path import absolute_path

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
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", absolute_path(requirements_path), "--no-cache-dir", "--target", absolute_path('.python_dependencies')], check=True, env=environ_copy, cwd=absolute_path("stable_diffusion/"))

class InstallDependencies(bpy.types.Operator):
    bl_idname = "stable_diffusion.install_dependencies"
    bl_label = "Install Dependencies"
    bl_description = ("Downloads and installs the required python packages into the '.python_dependencies' directory of the addon.")
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

        subprocess.run([sys.executable, absolute_path("stable_diffusion/scripts/preload_models.py")], check=True, cwd=absolute_path("stable_diffusion/"))

        return {"FINISHED"}