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
    "warning": "Requires installation of Stable Diffusion model weights",
    "blender": (3, 0, 0),
    "version": (0, 0, 6),
    "location": "",
    "warning": "",
    "category": "Paint"
}

import bpy
from bpy.props import IntProperty, PointerProperty, EnumProperty
import sys

from .help_section import register_section_props

from .prompt_engineering import *
from .operators.open_latest_version import check_for_updates
from .classes import CLASSES, PREFERENCE_CLASSES
from .tools import TOOLS
from .operators.dream_texture import kill_generator
from .property_groups.dream_prompt import DreamPrompt

requirements_path_items = (
    # Use the old version of requirements-win.txt to fix installation issues with Blender + PyTorch 1.12.1
    ('requirements-win-torch-1-11-0.txt', 'Linux/Windows (CUDA)', 'Linux or Windows with NVIDIA GPU'),
    ('stable_diffusion/requirements-mac-MPS-CPU.txt', 'Apple Silicon', 'Apple M1/M2'),
    ('stable_diffusion/requirements-lin-AMD.txt', 'Linux (AMD)', 'Linux with AMD GPU'),
)

def register():
    bpy.types.Scene.dream_textures_requirements_path = EnumProperty(name="Platform", items=requirements_path_items, description="Specifies which set of dependencies to install", default='stable_diffusion/requirements-mac-MPS-CPU.txt' if sys.platform == 'darwin' else 'requirements-win-torch-1-11-0.txt')
    
    register_section_props()

    for cls in PREFERENCE_CLASSES:
        bpy.utils.register_class(cls)

    check_for_updates()

    bpy.types.Scene.dream_textures_prompt = PointerProperty(type=DreamPrompt)
    bpy.types.Scene.init_img = PointerProperty(name="Init Image", type=bpy.types.Image)
    bpy.types.Scene.init_mask = PointerProperty(name="Init Mask", type=bpy.types.Image)
    bpy.types.Scene.dream_textures_history_selection = IntProperty()
    bpy.types.Scene.dream_textures_progress = bpy.props.IntProperty(name="Progress", default=0, min=0, max=0)
    bpy.types.Scene.dream_textures_info = bpy.props.StringProperty(name="Info")

    for cls in CLASSES:
        bpy.utils.register_class(cls)

    for tool in TOOLS:
        bpy.utils.register_tool(tool)

def unregister():
    for cls in PREFERENCE_CLASSES:
        bpy.utils.unregister_class(cls)

    for cls in CLASSES:
        bpy.utils.unregister_class(cls)
    for tool in TOOLS:
        bpy.utils.unregister_tool(tool)
    kill_generator()

if __name__ == "__main__":
    register()