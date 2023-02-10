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
    "author": "Dream Textures contributors",
    "description": "Use Stable Diffusion to generate unique textures straight from the shader editor.",
    "blender": (3, 1, 0),
    "version": (0, 1, 0),
    "location": "Image Editor -> Sidebar -> Dream",
    "category": "Paint"
}

from multiprocessing import current_process

if current_process().name != "__actor__":
    import bpy
    from bpy.props import IntProperty, PointerProperty, EnumProperty, BoolProperty, CollectionProperty, FloatProperty
    import sys
    import os

    module_name = os.path.basename(os.path.dirname(__file__))
    def clear_modules():
        for name in list(sys.modules.keys()):
            if name.startswith(module_name) and name != module_name:
                del sys.modules[name]
    clear_modules() # keep before all addon imports

    from .render_pass import register_render_pass, unregister_render_pass, pass_inputs
    from .prompt_engineering import *
    from .operators.open_latest_version import check_for_updates
    from .operators.project import framebuffer_arguments
    from .classes import CLASSES, PREFERENCE_CLASSES
    from .tools import TOOLS
    from .operators.dream_texture import DreamTexture, kill_generator
    from .property_groups.dream_prompt import DreamPrompt
    from .property_groups.seamless_result import SeamlessResult
    from .preferences import StableDiffusionPreferences
    from .ui.presets import register_default_presets

    requirements_path_items = (
        ('requirements/win-linux-cuda.txt', 'Linux/Windows (CUDA)', 'Linux or Windows with NVIDIA GPU'),
        ('requirements/mac-mps-cpu.txt', 'Apple Silicon', 'Apple M1/M2'),
        ('requirements/linux-rocm.txt', 'Linux (AMD)', 'Linux with AMD GPU'),
        ('requirements/win-dml.txt', 'Windows (DirectML)', 'Windows with DirectX 12 GPU'),
        ('requirements/dreamstudio.txt', 'DreamStudio', 'Cloud Compute Service')
    )

    def register():
        dt_op = bpy.ops
        for name in DreamTexture.bl_idname.split("."):
            dt_op = getattr(dt_op, name)
        if hasattr(bpy.types, dt_op.idname()): # objects under bpy.ops are created on the fly, have to check that it actually exists a little differently
            raise RuntimeError("Another instance of Dream Textures is already running.")

        bpy.types.Scene.dream_textures_requirements_path = EnumProperty(name="Platform", items=requirements_path_items, description="Specifies which set of dependencies to install", default='requirements/mac-mps-cpu.txt' if sys.platform == 'darwin' else 'requirements/win-linux-cuda.txt')

        for cls in PREFERENCE_CLASSES:
            bpy.utils.register_class(cls)
        
        bpy.types.Scene.dream_textures_history = CollectionProperty(type=DreamPrompt)

        check_for_updates()

        bpy.types.Scene.dream_textures_prompt = PointerProperty(type=DreamPrompt)
        bpy.types.Scene.dream_textures_prompt_file = PointerProperty(type=bpy.types.Text)
        bpy.types.Scene.init_img = PointerProperty(name="Init Image", type=bpy.types.Image)
        bpy.types.Scene.init_mask = PointerProperty(name="Init Mask", type=bpy.types.Image)
        bpy.types.Scene.init_depth = PointerProperty(name="Init Depth", type=bpy.types.Image, description="Use an existing depth map. Leave blank to generate one from the init image")
        bpy.types.Scene.seamless_result = PointerProperty(type=SeamlessResult)
        def get_selection_preview(self):
            history = bpy.context.scene.dream_textures_history
            if self.dream_textures_history_selection > 0 and self.dream_textures_history_selection < len(history):
                return history[self.dream_textures_history_selection].generate_prompt()
            return ""
        bpy.types.Scene.dream_textures_history_selection = IntProperty(default=1)
        bpy.types.Scene.dream_textures_history_selection_preview = bpy.props.StringProperty(name="", default="", get=get_selection_preview, set=lambda _, __: None)
        bpy.types.Scene.dream_textures_progress = bpy.props.IntProperty(name="", default=0, min=0, max=0)
        bpy.types.Scene.dream_textures_info = bpy.props.StringProperty(name="Info")

        bpy.types.Scene.dream_textures_viewport_enabled = BoolProperty(name="Viewport Enabled", default=False)
        bpy.types.Scene.dream_textures_render_properties_enabled = BoolProperty(default=False)
        bpy.types.Scene.dream_textures_render_properties_prompt = PointerProperty(type=DreamPrompt)
        bpy.types.Scene.dream_textures_render_properties_pass_inputs = EnumProperty(name="Pass Inputs", items=pass_inputs)
        
        bpy.types.Scene.dream_textures_upscale_prompt = PointerProperty(type=DreamPrompt)
        bpy.types.Scene.dream_textures_upscale_tile_size = IntProperty(name="Tile Size", default=128, step=64, min=64, max=512)
        bpy.types.Scene.dream_textures_upscale_blend = IntProperty(name="Blend", default=32, step=8, min=0, max=512)
        bpy.types.Scene.dream_textures_upscale_seamless_result = PointerProperty(type=SeamlessResult)
        
        bpy.types.Scene.dream_textures_project_prompt = PointerProperty(type=DreamPrompt)
        bpy.types.Scene.dream_textures_project_framebuffer_arguments = EnumProperty(name="Inputs", items=framebuffer_arguments)
        bpy.types.Scene.dream_textures_project_bake = BoolProperty(name="Bake", default=False, description="Re-maps the generated texture onto the specified UV map")

        for cls in CLASSES:
            bpy.utils.register_class(cls)

        for tool in TOOLS:
            bpy.utils.register_tool(tool)

        # Monkey patch cycles render passes
        register_render_pass()

        register_default_presets()

    def unregister():
        for cls in PREFERENCE_CLASSES:
            bpy.utils.unregister_class(cls)

        for cls in CLASSES:
            bpy.utils.unregister_class(cls)
        for tool in TOOLS:
            bpy.utils.unregister_tool(tool)
        
        unregister_render_pass()

        kill_generator()