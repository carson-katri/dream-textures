import bpy
import os
import site
import sys
import sysconfig
import subprocess
import requests
import tarfile
from enum import IntEnum

from ..absolute_path import absolute_path
from ..generator_process import Generator

class PipInstall(IntEnum):
    DEPENDENCIES = 1
    STANDARD = 2
    USER_SITE = 3

def install_pip(method = PipInstall.STANDARD):
    """
    Installs pip if not already present. Please note that ensurepip.bootstrap() also calls pip, which adds the
    environment variable PIP_REQ_TRACKER. After ensurepip.bootstrap() finishes execution, the directory doesn't exist
    anymore. However, when subprocess is used to call pip, in order to install a package, the environment variables
    still contain PIP_REQ_TRACKER with the now nonexistent path. This is a problem since pip checks if PIP_REQ_TRACKER
    is set and if it is, attempts to use it as temp directory. This would result in an error because the
    directory can't be found. Therefore, PIP_REQ_TRACKER needs to be removed from environment variables.
    :return:
    """

    import ensurepip

    if method == PipInstall.DEPENDENCIES:
        # ensurepip doesn't have a useful way of installing to a specific directory.
        # root parameter can be used, but it just concatenates that to the beginning of
        # where it decides to install to, causing a more complicated path to where it installs.
        wheels = {}
        for name, package in ensurepip._get_packages().items():
            if package.wheel_name:
                whl = os.path.join(os.path.dirname(ensurepip.__file__), "_bundled", package.wheel_name)
            else:
                whl = package.wheel_path
            wheels[name] = whl
        pip_whl = os.path.join(wheels['pip'], 'pip')
        subprocess.run([sys.executable, pip_whl, "install", *wheels.values(), "--upgrade", "--no-index", "--no-deps", "--no-cache-dir", "--target", absolute_path(".python_dependencies")], check=True)
        return
    
    # STANDARD or USER_SITE
    no_user = os.environ.get("PYTHONNOUSERSITE", None)
    if method == PipInstall.STANDARD:
        os.environ["PYTHONNOUSERSITE"] = "1"
    else:
        os.environ.pop("PYTHONNOUSERSITE", None)
    try:
        ensurepip.bootstrap(user=method==PipInstall.USER_SITE)
    finally:
        os.environ.pop("PIP_REQ_TRACKER", None)
        if no_user:
            os.environ["PYTHONNOUSERSITE"] = no_user
        else:
            os.environ.pop("PYTHONNOUSERSITE", None)

def install_pip_any(*methods):
    methods = methods or PipInstall
    for method in methods:
        print(f"Attempting to install pip: {PipInstall(method).name}")
        try:
            install_pip(method)
            return method
        except:
            import traceback
            traceback.print_exc()

def get_pip_install():
    def run(pip):
        if os.path.exists(pip):
            try:
                subprocess.run([sys.executable, pip, "--version"], check=True)
                return True
            except subprocess.CalledProcessError:
                pass
        return False

    if run(absolute_path(".python_dependencies/pip")):
        return PipInstall.DEPENDENCIES
    
    # This seems to not raise CalledProcessError while debugging in vscode, but works fine in normal use.
    # subprocess.run([sys.executable, "-s", "-m", "pip", "--version"], check=True)
    # Best to check if the module directory exists first.
    for path in site.getsitepackages():
        if run(os.path.join(path,"pip")):
            return PipInstall.STANDARD

    if run(os.path.join(site.getusersitepackages(),"pip")):
        return PipInstall.USER_SITE


def install_and_import_requirements(requirements_txt=None, pip_install=PipInstall.STANDARD):
    """
    Installs all modules in the 'requirements.txt' file.
    """
    environ_copy = dict(os.environ)
    if pip_install != PipInstall.USER_SITE:
        environ_copy["PYTHONNOUSERSITE"] = "1"
    if pip_install == PipInstall.DEPENDENCIES:
        environ_copy["PYTHONPATH"] = absolute_path(".python_dependencies")
    python_include_dir = sysconfig.get_paths()['include']
    if not os.path.exists(python_include_dir):
        try:
            os.makedirs(python_include_dir)
        finally:
            pass
    if os.access(python_include_dir, os.W_OK):
        print("downloading additional include files")
        python_devel_tgz_path = absolute_path('python-devel.tgz')
        response = requests.get(f"https://www.python.org/ftp/python/{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}/Python-{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}.tgz")
        with open(python_devel_tgz_path, 'wb') as f:
            f.write(response.content)
        with tarfile.open(python_devel_tgz_path) as python_devel_tgz:
            def members(tf):
                prefix = f"Python-{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}/Include/"
                l = len(prefix)
                for member in tf.getmembers():
                    if member.path.startswith(prefix):
                        member.path = member.path[l:]
                        yield member
            python_devel_tgz.extractall(path=python_include_dir, members=members(python_devel_tgz))
        os.remove(python_devel_tgz_path)
    else:
        print(f"skipping include files, can't write to {python_include_dir}",file=sys.stderr)

    subprocess.run([sys.executable, "-m", "pip", "install", "-r", absolute_path(requirements_txt), "--upgrade", "--no-cache-dir", "--target", absolute_path('.python_dependencies')], check=True, env=environ_copy, cwd=absolute_path(""))

class InstallDependencies(bpy.types.Operator):
    bl_idname = "stable_diffusion.install_dependencies"
    bl_label = "Install Dependencies"
    bl_description = ("Downloads and installs the required python packages into the '.python_dependencies' directory of the addon.")
    bl_options = {"REGISTER", "INTERNAL"}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        # Open the console so we can watch the progress.
        if sys.platform == 'win32':
            bpy.ops.wm.console_toggle()

        Generator.shared_close()
        try:
            pip_install = get_pip_install()
            if pip_install is None:
                pip_install = install_pip_any()
            if pip_install is None:
                raise ImportError(f'Pip could not be installed. You may have to manually install pip into {absolute_path(".python_dependencies")}')

            install_and_import_requirements(requirements_txt=context.scene.dream_textures_requirements_path, pip_install=pip_install)
        except (subprocess.CalledProcessError, ImportError) as err:
            self.report({"ERROR"}, str(err))
            return {"CANCELLED"}

        return {"FINISHED"}

class UninstallDependencies(bpy.types.Operator):
    bl_idname = "stable_diffusion.uninstall_dependencies"
    bl_label = "Uninstall Dependencies"
    bl_description = ("Uninstalls specific dependencies from Blender's site-packages")
    bl_options = {"REGISTER", "INTERNAL"}

    conflicts: bpy.props.StringProperty(name="Conflicts")

    def execute(self, context):
        # Open the console so we can watch the progress.
        if sys.platform == 'win32':
            bpy.ops.wm.console_toggle()

        environ_copy = dict(os.environ)
        environ_copy["PYTHONNOUSERSITE"] = "1"
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", *self.conflicts.split(' ')], check=True, env=environ_copy, cwd=absolute_path(""))

        return {"FINISHED"}