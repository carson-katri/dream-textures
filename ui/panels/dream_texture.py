import bpy
from bpy.types import Panel
from ..presets import DREAM_PT_AdvancedPresets
from ...prompt_engineering import *
from ...operators.dream_texture import DreamTexture, ReleaseGenerator, CancelGenerator, get_source_image
from ...operators.open_latest_version import OpenLatestVersion, is_force_show_download, new_version_available
from ...operators.view_history import ImportPromptFile
from ..space_types import SPACE_TYPES
from ...generator_process.actions.detect_seamless import SeamlessAxes
from ...api.models import FixItError
from ...property_groups.dream_prompt import DreamPrompt
from ...property_groups.control_net import BakeControlNetImage
from ... import api

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

                layout.prop(context.scene.dream_textures_prompt, "backend")
                layout.prop(context.scene.dream_textures_prompt, 'model')

        DreamTexturePanel.__name__ = f"DREAM_PT_dream_panel_{space_type}"
        yield DreamTexturePanel

        def get_prompt(context):
            return context.scene.dream_textures_prompt

        def get_seamless_result(context, prompt):
            init_image = None
            if prompt.use_init_img and prompt.init_img_action in ['modify', 'inpaint']:
                init_image = get_source_image(context, prompt.init_img_src)
            context.scene.seamless_result.check(init_image)
            return context.scene.seamless_result

        yield from create_panel(space_type, 'UI', DreamTexturePanel.bl_idname, prompt_panel, get_prompt,
                                get_seamless_result=get_seamless_result)
        yield create_panel(space_type, 'UI', DreamTexturePanel.bl_idname, size_panel, get_prompt)
        yield from create_panel(space_type, 'UI', DreamTexturePanel.bl_idname, init_image_panels, get_prompt)
        yield create_panel(space_type, 'UI', DreamTexturePanel.bl_idname, control_net_panel, get_prompt)
        yield from create_panel(space_type, 'UI', DreamTexturePanel.bl_idname, advanced_panel, get_prompt)
        yield create_panel(space_type, 'UI', DreamTexturePanel.bl_idname, actions_panel, get_prompt)

def create_panel(space_type, region_type, parent_id, ctor, get_prompt, use_property_decorate=False, **kwargs):
    class BasePanel(Panel):
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

    return ctor(kwargs.pop('base_panel', SubPanel), space_type, get_prompt, **kwargs)

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
            return get_prompt(context).prompt_structure != file_batch_structure.id

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
            if prompt.init_img_action != 'outpaint':
                layout.prop(prompt, "strength")
            layout.prop(prompt, "use_init_img_color")
            if prompt.init_img_action == 'modify':
                layout.prop(prompt, "modify_action_source_type")
                if prompt.modify_action_source_type == 'depth_map':
                    layout.template_ID(context.scene, "init_depth", open="image.open")
    yield InitImagePanel

def control_net_panel(sub_panel, space_type, get_prompt):
    class ControlNetPanel(sub_panel):
        """Create a subpanel for ControlNet options"""
        bl_idname = f"DREAM_PT_dream_panel_control_net_{space_type}"
        bl_label = "ControlNet"
        bl_options = {'DEFAULT_CLOSED'}

        def draw(self, context):
            layout = self.layout
            prompt = get_prompt(context)
            
            layout.operator("wm.call_menu", text="Add ControlNet", icon='ADD').name = "DREAM_MT_control_nets_add"
            for i, control_net in enumerate(prompt.control_nets):
                box = layout.box()
                box.use_property_split = False
                box.use_property_decorate = False
                
                row = box.row()
                row.prop(control_net, "enabled", icon="MODIFIER_ON" if control_net.enabled else "MODIFIER_OFF", icon_only=True, emboss=False)
                row.prop(control_net, "control_net", text="")
                row.operator("dream_textures.control_nets_remove", icon='X', emboss=False, text="").index = i

                col = box.column()
                col.use_property_split = True
                col.template_ID(control_net, "control_image", open="image.open", text="Image")
                processor_row = col.row()
                processor_row.prop(control_net, "processor_id")
                if control_net.processor_id != "none":
                    processor_row.operator(BakeControlNetImage.bl_idname, icon='RENDER_STILL', text='').index = i
                col.prop(control_net, "conditioning_scale")
                
    return ControlNetPanel

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
            
            prompt = get_prompt(context)
            layout.prop(prompt, "random_seed")
            if not prompt.random_seed:
                layout.prop(prompt, "seed")
            # advanced_box.prop(self, "iterations") # Disabled until supported by the addon.
            layout.prop(prompt, "steps")
            layout.prop(prompt, "cfg_scale")
            layout.prop(prompt, "scheduler")
            layout.prop(prompt, "step_preview_mode")

            backend: api.Backend = prompt.get_backend()
            backend.draw_advanced(layout, context)

    yield AdvancedPanel

    yield from optimization_panels(sub_panel, space_type, get_prompt, AdvancedPanel.bl_idname)

def optimization_panels(sub_panel, space_type, get_prompt, parent_id=""):
    class SpeedOptimizationPanel(sub_panel):
        """Create a subpanel for speed optimizations"""
        bl_idname = f"DREAM_PT_dream_panel_speed_optimizations_{space_type}"
        bl_label = "Speed Optimizations"
        bl_parent_id = parent_id

        def draw(self, context):
            super().draw(context)
            layout = self.layout
            layout.use_property_split = True
            prompt = get_prompt(context)

            backend: api.Backend = prompt.get_backend()
            backend.draw_speed_optimizations(layout, context)
    yield SpeedOptimizationPanel

    class MemoryOptimizationPanel(sub_panel):
        """Create a subpanel for memory optimizations"""
        bl_idname = f"DREAM_PT_dream_panel_memory_optimizations_{space_type}"
        bl_label = "Memory Optimizations"
        bl_parent_id = parent_id

        def draw(self, context):
            super().draw(context)
            layout = self.layout
            layout.use_property_split = True
            prompt = get_prompt(context)

            backend: api.Backend = prompt.get_backend()
            backend.draw_memory_optimizations(layout, context)
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
            
            row = layout.row(align=True)
            row.scale_y = 1.5
            if CancelGenerator.poll(context):
                row.operator(CancelGenerator.bl_idname, icon="SNAP_FACE", text="")
            if context.scene.dream_textures_progress <= 0:
                if context.scene.dream_textures_info != "":
                    disabled_row = row.row(align=True)
                    disabled_row.operator(DreamTexture.bl_idname, text=context.scene.dream_textures_info, icon="INFO")
                    disabled_row.enabled = False
                else:
                    row.operator(DreamTexture.bl_idname, icon="PLAY", text="Generate")
            else:
                if bpy.app.version[0] >= 4:
                    progress = context.scene.dream_textures_progress
                    progress_max = bpy.types.Scene.dream_textures_progress.keywords['max']
                    row.progress(text=f"{progress} / {progress_max}", factor=progress / progress_max)
                else:
                    disabled_row = row.row(align=True)
                    disabled_row.use_property_split = True
                    disabled_row.prop(context.scene, 'dream_textures_progress', slider=True)
                    disabled_row.enabled = False
            row.operator(ReleaseGenerator.bl_idname, icon="X", text="")

            if context.scene.dream_textures_last_execution_time != "":
                r = layout.row()
                r.scale_x = 0.5
                r.scale_y = 0.5
                r.label(text=context.scene.dream_textures_last_execution_time, icon="SORTTIME")

            # Validation
            try:
                backend: api.Backend = prompt.get_backend()
                backend.validate(prompt.generate_args(context))
            except FixItError as e:
                error_box = layout.box()
                error_box.use_property_split = False
                for i, line in enumerate(e.args[0].split('\n')):
                    error_box.label(text=line, icon="ERROR" if i == 0 else "NONE")
                e._draw(prompt, context, error_box)
    return ActionsPanel