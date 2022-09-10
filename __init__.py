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
    "version": (0, 0, 4),
    "location": "",
    "warning": "",
    "category": "Node"
}

import bpy
from bpy.props import StringProperty, EnumProperty, PointerProperty
import sys
import importlib

from .async_loop import *
from .prompt_engineering import *
from .operators.open_latest_version import check_for_updates
from .absolute_path import absolute_path
from .classes import CLASSES, PREFERENCE_CLASSES
from .shader_menu import shader_menu_draw
from .operators.install_dependencies import are_dependencies_installed, set_dependencies_installed

def register():
    async_loop.setup_asyncio_executor()
    bpy.utils.register_class(AsyncLoopModalOperator)

    sys.path.append(absolute_path("stable_diffusion/"))
    sys.path.append(absolute_path("stable_diffusion/src/clip"))
    sys.path.append(absolute_path("stable_diffusion/src/k-diffusion"))
    sys.path.append(absolute_path("stable_diffusion/src/taming-transformers"))

    set_dependencies_installed(False)

    bpy.types.Scene.init_img = PointerProperty(name="Init Image", type=bpy.types.Image)

    def map_structure_token_items(value):
        return (value[0], value[1], '')
    for structure in prompt_structures:
        for token in structure.structure:
            if not isinstance(token, str) and not hasattr(bpy.types.Scene, token.id):
                setattr(bpy.types.Scene, 'prompt_structure_token_' + token.id, StringProperty(name=token.label))
                setattr(bpy.types.Scene, 'prompt_structure_token_' + token.id + '_enum', EnumProperty(
                    name=token.label,
                    items=[('custom', 'Custom', '')] + list(map(map_structure_token_items, token.values)),
                    default='custom' if len(token.values) == 0 else token.values[0][0],
                ))

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
    
    bpy.types.NODE_HT_header.append(shader_menu_draw)

def unregister():
    bpy.utils.unregister_class(AsyncLoopModalOperator)

    for cls in PREFERENCE_CLASSES:
        bpy.utils.unregister_class(cls)

    if are_dependencies_installed():
        for cls in CLASSES:
            bpy.utils.unregister_class(cls)
        bpy.types.NODE_HT_header.remove(shader_menu_draw)


if __name__ == "__main__":
    register()