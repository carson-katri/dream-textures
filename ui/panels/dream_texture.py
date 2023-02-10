from bpy.types import Panel
from bpy_extras.io_utils import ImportHelper

import webbrowser
import os
import shutil

from ...absolute_path import CLIPSEG_WEIGHTS_PATH
from ..presets import DREAM_PT_AdvancedPresets
from ...pil_to_image import *
from ...prompt_engineering import *
from ...operators.dream_texture import DreamTexture, ReleaseGenerator, CancelGenerator
from ...operators.open_latest_version import OpenLatestVersion, is_force_show_download, new_version_available
from ...operators.view_history import ImportPromptFile
from ..space_types import SPACE_TYPES
from ...property_groups.dream_prompt import DreamPrompt, pipeline_options
from ...generator_process.actions.prompt_to_image import Optimizations
from ...generator_process.actions.detect_seamless import SeamlessAxes
from ...generator_process.models import Pipeline, FixItError

def dream_texture_panels():
    for space_type in SPACE_TYPES:
        class DreamTexturePanel(Panel):
            """Creates a Panel in the scene context of the properties editor"""
            bl_label = "Dream Texture"
            bl_idname = f"DREAM_PT_dream_panel_{space_type}"
            bl_category = "Dream"
            bl_space_type = space_type
            bl_region_type = 'UI'

            @classmethod
            def poll(cls, context):
                if cls.bl_space_type == 'NODE_EDITOR':
                    return context.area.ui_type == "ShaderNodeTree" or context.area.ui_type == "CompositorNodeTree"
                else:
                    return True
            
            def draw_header_preset(self, context):
                layout = self.layout
                layout.operator(ImportPromptFile.bl_idname, text="", icon="IMPORT")
                layout.separator()

            def draw(self, context):
                layout = self.layout
                layout.use_property_split = True
                layout.use_property_decorate = False

                if is_force_show_download():
                    layout.operator(OpenLatestVersion.bl_idname, icon="IMPORT", text="Download Latest Release")
                elif new_version_available():
                    layout.operator(OpenLatestVersion.bl_idname, icon="IMPORT")

                layout.prop(context.scene.dream_textures_prompt, "pipeline")
                if Pipeline[context.scene.dream_textures_prompt.pipeline].model():
                    layout.prop(context.scene.dream_textures_prompt, 'model')

        DreamTexturePanel.__name__ = f"DREAM_PT_dream_panel_{space_type}"
        yield DreamTexturePanel

        def get_prompt(context):
            return context.scene.dream_textures_prompt

        def get_seamless_result(context, prompt):
            init_image = None
            if prompt.use_init_img and prompt.init_img_action in ['modify', 'inpaint']:
                match prompt.init_img_src:
                    case 'file':
                        init_image = context.scene.init_img
                    case 'open_editor':
                        for area in context.screen.areas:
                            if area.type == 'IMAGE_EDITOR':
                                if area.spaces.active.image is not None:
                                    init_image = area.spaces.active.image
            context.scene.seamless_result.check(init_image)
            return context.scene.seamless_result

        yield from create_panel(space_type, 'UI', DreamTexturePanel.bl_idname, prompt_panel, get_prompt,
                                get_seamless_result=get_seamless_result)
        yield create_panel(space_type, 'UI', DreamTexturePanel.bl_idname, size_panel, get_prompt)
        yield from create_panel(space_type, 'UI', DreamTexturePanel.bl_idname, init_image_panels, get_prompt)
        yield from create_panel(space_type, 'UI', DreamTexturePanel.bl_idname, advanced_panel, get_prompt)
        yield create_panel(space_type, 'UI', DreamTexturePanel.bl_idname, actions_panel, get_prompt)

def create_panel(space_type, region_type, parent_id, ctor, get_prompt, use_property_decorate=False, **kwargs):
    class BasePanel(bpy.types.Panel):
        bl_category = "Dream"
        bl_space_type = space_type
        bl_region_type = region_type

    class SubPanel(BasePanel):
        bl_category = "Dream"
        bl_space_type = space_type
        bl_region_type = region_type
        bl_parent_id = parent_id

        def draw(self, context):
            self.layout.use_property_decorate = use_property_decorate
    
    return ctor(SubPanel, space_type, get_prompt, **kwargs)

def prompt_panel(sub_panel, space_type, get_prompt, get_seamless_result=None):
    class PromptPanel(sub_panel):
        """Create a subpanel for prompt input"""
        bl_label = "Prompt"
        bl_idname = f"DREAM_PT_dream_panel_prompt_{space_type}"

        def draw_header_preset(self, context):
            layout = self.layout
            layout.prop(get_prompt(context), "prompt_structure", text="")

        def draw(self, context):
            super().draw(context)
            layout = self.layout
            layout.use_property_split = True
            prompt = get_prompt(context)

            structure = next(x for x in prompt_structures if x.id == prompt.prompt_structure)
            for segment in structure.structure:
                segment_row = layout.row()
                enum_prop = 'prompt_structure_token_' + segment.id + '_enum'
                is_custom = getattr(prompt, enum_prop) == 'custom'
                if is_custom:
                    segment_row.prop(prompt, 'prompt_structure_token_' + segment.id)
                enum_cases = DreamPrompt.__annotations__[enum_prop].keywords['items']
                if len(enum_cases) != 1 or enum_cases[0][0] != 'custom':
                    segment_row.prop(prompt, enum_prop, icon_only=is_custom)
            if prompt.prompt_structure == file_batch_structure.id:
                layout.template_ID(context.scene, "dream_textures_prompt_file", open="text.open")
            if Pipeline[prompt.pipeline].seamless():
                layout.prop(prompt, "seamless_axes")
                if prompt.seamless_axes == SeamlessAxes.AUTO and get_seamless_result is not None:
                    auto_row = self.layout.row()
                    auto_row.enabled = False
                    auto_row.prop(get_seamless_result(context, prompt), "result")

    yield PromptPanel

    class NegativePromptPanel(sub_panel):
        """Create a subpanel for negative prompt input"""
        bl_idname = f"DREAM_PT_dream_panel_negative_prompt_{space_type}"
        bl_label = "Negative"
        bl_parent_id = PromptPanel.bl_idname

        @classmethod
        def poll(cls, context):
            return get_prompt(context).prompt_structure != file_batch_structure.id and Pipeline[get_prompt(context).pipeline].negative_prompts()

        def draw_header(self, context):
            layout = self.layout
            layout.prop(get_prompt(context), "use_negative_prompt", text="")

        def draw(self, context):
            super().draw(context)
            layout = self.layout
            layout.use_property_split = True
            layout.enabled = layout.enabled and get_prompt(context).use_negative_prompt
            scene = context.scene

            layout.prop(get_prompt(context), "negative_prompt")
    yield NegativePromptPanel

def size_panel(sub_panel, space_type, get_prompt):
    class SizePanel(sub_panel):
        """Create a subpanel for size options"""
        bl_idname = f"DREAM_PT_dream_panel_size_{space_type}"
        bl_label = "Size"
        bl_options = {'DEFAULT_CLOSED'}

        def draw_header(self, context):
            self.layout.prop(get_prompt(context), "use_size", text="")

        def draw(self, context):
            super().draw(context)
            layout = self.layout
            layout.use_property_split = True
            layout.enabled = layout.enabled and get_prompt(context).use_size

            layout.prop(get_prompt(context), "width")
            layout.prop(get_prompt(context), "height")
    return SizePanel

def init_image_panels(sub_panel, space_type, get_prompt):
    class InitImagePanel(sub_panel):
        """Create a subpanel for init image options"""
        bl_idname = f"DREAM_PT_dream_panel_init_image_{space_type}"
        bl_label = "Source Image"
        bl_options = {'DEFAULT_CLOSED'}

        def draw_header(self, context):
            self.layout.prop(get_prompt(context), "use_init_img", text="")

        def draw(self, context):
            super().draw(context)
            layout = self.layout
            prompt = get_prompt(context)
            layout.enabled = prompt.use_init_img
            
            layout.prop(prompt, "init_img_src", expand=True)
            if prompt.init_img_src == 'file':
                layout.template_ID(context.scene, "init_img", open="image.open")
            layout.prop(prompt, "init_img_action", expand=True)
            
            layout.use_property_split = True

            if prompt.init_img_action == 'inpaint':
                layout.prop(prompt, "inpaint_mask_src")
                if prompt.inpaint_mask_src == 'prompt':
                    layout.prop(prompt, "text_mask")
                    layout.prop(prompt, "text_mask_confidence")
                layout.prop(prompt, "inpaint_replace")
            elif prompt.init_img_action == 'outpaint':
                layout.prop(prompt, "outpaint_origin")
                def _outpaint_warning_box(warning):
                    box = layout.box()
                    box.label(text=warning, icon="ERROR")
                if prompt.outpaint_origin[0] <= -prompt.width or prompt.outpaint_origin[1] <= -prompt.height:
                    _outpaint_warning_box("Outpaint has no overlap, so the result will not blend")
                init_img = context.scene.init_img if prompt.init_img_src == 'file' else None
                if init_img is None:
                    for area in context.screen.areas:
                        if area.type == 'IMAGE_EDITOR':
                            if area.spaces.active.image is not None:
                                init_img = area.spaces.active.image
                if init_img is not None:
                    if prompt.outpaint_origin[0] >= init_img.size[0] or \
                        prompt.outpaint_origin[1] >= init_img.size[1]:
                        _outpaint_warning_box("Outpaint has no overlap, so the result will not blend")
            elif prompt.init_img_action == 'modify':
                layout.prop(prompt, "fit")
            layout.prop(prompt, "strength")
            if Pipeline[prompt.pipeline].color_correction():
                layout.prop(prompt, "use_init_img_color")
            if prompt.init_img_action == 'modify':
                layout.prop(prompt, "modify_action_source_type")
                if prompt.modify_action_source_type == 'depth_map':
                    layout.template_ID(context.scene, "init_depth", open="image.open")
    yield InitImagePanel

def advanced_panel(sub_panel, space_type, get_prompt):
    class AdvancedPanel(sub_panel):
        """Create a subpanel for advanced options"""
        bl_idname = f"DREAM_PT_dream_panel_advanced_{space_type}"
        bl_label = "Advanced"
        bl_options = {'DEFAULT_CLOSED'}

        def draw_header_preset(self, context):
            DREAM_PT_AdvancedPresets.draw_panel_header(self.layout)

        def draw(self, context):
            super().draw(context)
            layout = self.layout
            layout.use_property_split = True
            
            layout.prop(get_prompt(context), "random_seed")
            if not get_prompt(context).random_seed:
                layout.prop(get_prompt(context), "seed")
            # advanced_box.prop(self, "iterations") # Disabled until supported by the addon.
            layout.prop(get_prompt(context), "steps")
            layout.prop(get_prompt(context), "cfg_scale")
            layout.prop(get_prompt(context), "scheduler")
            layout.prop(get_prompt(context), "step_preview_mode")

    yield AdvancedPanel

    class SpeedOptimizationPanel(sub_panel):
        """Create a subpanel for speed optimizations"""
        bl_idname = f"DREAM_PT_dream_panel_speed_optimizations_{space_type}"
        bl_label = "Speed Optimizations"
        bl_parent_id = AdvancedPanel.bl_idname

        def draw(self, context):
            super().draw(context)
            layout = self.layout
            layout.use_property_split = True
            prompt = get_prompt(context)

            inferred_device = Optimizations.infer_device()
            def optimization(prop):
                if hasattr(prompt, f"optimizations_{prop}"):
                    if Optimizations().can_use(prop, inferred_device):
                        layout.prop(prompt, f"optimizations_{prop}")

            optimization("cudnn_benchmark")
            optimization("tf32")
            optimization("amp")
            optimization("half_precision")
            optimization("channels_last_memory_format")
            optimization("batch_size")
    yield SpeedOptimizationPanel

    class MemoryOptimizationPanel(sub_panel):
        """Create a subpanel for memory optimizations"""
        bl_idname = f"DREAM_PT_dream_panel_memory_optimizations_{space_type}"
        bl_label = "Memory Optimizations"
        bl_parent_id = AdvancedPanel.bl_idname

        def draw(self, context):
            super().draw(context)
            layout = self.layout
            layout.use_property_split = True
            prompt = get_prompt(context)

            def optimization(prop):
                if hasattr(prompt, f"optimizations_{prop}"):
                    layout.prop(prompt, f"optimizations_{prop}")

            optimization("attention_slicing")
            slice_size_row = layout.row()
            slice_size_row.prop(prompt, "optimizations_attention_slice_size_src")
            if prompt.optimizations_attention_slice_size_src == 'manual':
                slice_size_row.prop(prompt, "optimizations_attention_slice_size", text="Size")
            optimization("sequential_cpu_offload")
            optimization("cpu_only")
            # optimization("xformers_attention") # FIXME: xFormers is not yet available.
            optimization("vae_slicing")
    yield MemoryOptimizationPanel

def actions_panel(sub_panel, space_type, get_prompt):
    class ActionsPanel(sub_panel):
        """Create a subpanel for actions"""
        bl_idname = f"DREAM_PT_dream_panel_actions_{space_type}"
        bl_label = "Advanced"
        bl_options = {'HIDE_HEADER'}

        def draw(self, context):
            super().draw(context)
            layout = self.layout
            layout.use_property_split = True

            prompt = get_prompt(context)

            iterations_row = layout.row()
            iterations_row.enabled = prompt.prompt_structure != file_batch_structure.id
            iterations_row.prop(prompt, "iterations")
            
            row = layout.row()
            row.scale_y = 1.5
            if context.scene.dream_textures_progress <= 0:
                if context.scene.dream_textures_info != "":
                    row.label(text=context.scene.dream_textures_info, icon="INFO")
                else:
                    row.operator(DreamTexture.bl_idname, icon="PLAY", text="Generate")
            else:
                disabled_row = row.row()
                disabled_row.use_property_split = True
                disabled_row.prop(context.scene, 'dream_textures_progress', slider=True)
                disabled_row.enabled = False
            if CancelGenerator.poll(context):
                row.operator(CancelGenerator.bl_idname, icon="CANCEL", text="")
            row.operator(ReleaseGenerator.bl_idname, icon="X", text="")

            # Validation
            try:
                prompt.validate(context)
            except FixItError as e:
                error_box = layout.box()
                error_box.use_property_split = False
                for i, line in enumerate(e.args[0].split('\n')):
                    error_box.label(text=line, icon="ERROR" if i == 0 else "NONE")
                e.draw(context, error_box)
            except Exception as e:
                print(e)
    return ActionsPanel
