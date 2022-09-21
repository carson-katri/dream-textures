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
    "author": "Carson Katri, Greg Richardson, Kevin C. Burke",
    "description": "Use Stable Diffusion to generate unique textures straight from the shader editor.",
    "warning": "Requires installation of dependencies",
    "blender": (2, 80, 0),
    "version": (0, 0, 5),
    "location": "",
    "warning": "",
    "category": "Node"
}

import bpy
from bpy.props import IntProperty, PointerProperty, EnumProperty
import sys
import importlib

from .help_section import register_section_props

from .async_loop import *
from .prompt_engineering import *
from .operators.open_latest_version import check_for_updates
from .absolute_path import absolute_path
from .classes import CLASSES, PREFERENCE_CLASSES
from .tools import TOOLS
from .operators.install_dependencies import are_dependencies_installed, set_dependencies_installed
from .property_groups.dream_prompt import DreamPrompt
from .ui import panel

requirements_path_items = (
    # Use the old version of requirements-win.txt to fix installation issues with Blender + PyTorch 1.12.1
    ('requirements-win-torch-1-11-0.txt', 'Linux/Windows (CUDA)', 'Linux or Windows with NVIDIA GPU'),
    ('stable_diffusion/requirements-mac-MPS-CPU.txt', 'Apple Silicon', 'Apple M1/M2'),
    ('stable_diffusion/requirements-lin-AMD.txt', 'Linux (AMD)', 'Linux with AMD GPU'),
)

def register():
    async_loop.setup_asyncio_executor()
    bpy.utils.register_class(AsyncLoopModalOperator)

    sys.path.append(absolute_path("stable_diffusion/"))
    sys.path.append(absolute_path("stable_diffusion/src/clip"))
    sys.path.append(absolute_path("stable_diffusion/src/k-diffusion"))
    sys.path.append(absolute_path("stable_diffusion/src/taming-transformers"))

    set_dependencies_installed(False)
    bpy.types.Scene.dream_textures_requirements_path = EnumProperty(name="Platform", items=requirements_path_items, description="Specifies which set of dependencies to install", default='stable_diffusion/requirements-mac-MPS-CPU.txt' if sys.platform == 'darwin' else 'requirements-win-torch-1-11-0.txt')
    
    register_section_props()

    for cls in PREFERENCE_CLASSES:
        bpy.utils.register_class(cls)

    check_for_updates()

    try:
        # Check if the last dependency is installed.
        importlib.import_module("transformers")
        set_dependencies_installed(True)
    except ModuleNotFoundError:
        # Don't register other panels, operators etc.
        return

    for cls in CLASSES:
        bpy.utils.register_class(cls)

    for tool in TOOLS:
        bpy.utils.register_tool(tool)

    bpy.types.Scene.dream_textures_prompt = PointerProperty(type=DreamPrompt)
    bpy.types.Scene.init_img = PointerProperty(name="Init Image", type=bpy.types.Image)
    bpy.types.Scene.init_mask = PointerProperty(name="Init Mask", type=bpy.types.Image)
    bpy.types.Scene.dream_textures_history_selection = IntProperty()

    panel.register()

def unregister():
    bpy.utils.unregister_class(AsyncLoopModalOperator)

    for cls in PREFERENCE_CLASSES:
        bpy.utils.unregister_class(cls)

    if are_dependencies_installed():
        for cls in CLASSES:
            bpy.utils.unregister_class(cls)
        for tool in TOOLS:
            bpy.utils.unregister_tool(tool)

    panel.unregister()

if __name__ == "__main__":
    register()