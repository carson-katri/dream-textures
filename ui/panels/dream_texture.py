from bpy.types import Panel
from ..presets import DREAM_PT_AdvancedPresets
from ...pil_to_image import *
from ...prompt_engineering import *
from ...operators.dream_texture import DreamTexture, ReleaseGenerator, CancelGenerator
from ...operators.open_latest_version import OpenLatestVersion, is_force_show_download, new_version_available
from ...operators.view_history import ImportPromptFile
from ..space_types import SPACE_TYPES
from ...property_groups.dream_prompt import DreamPrompt

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
            def poll(self, context):
                if self.bl_space_type == 'NODE_EDITOR':
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

                if is_force_show_download():
                    layout.operator(OpenLatestVersion.bl_idname, icon="IMPORT", text="Download Latest Release")
                elif new_version_available():
                    layout.operator(OpenLatestVersion.bl_idname, icon="IMPORT")

        DreamTexturePanel.__name__ = f"DREAM_PT_dream_panel_{space_type}"
        yield DreamTexturePanel

        def get_prompt(context):
            return context.scene.dream_textures_prompt
        yield from create_panel(space_type, 'UI', DreamTexturePanel.bl_idname, prompt_panel, get_prompt)
        yield create_panel(space_type, 'UI', DreamTexturePanel.bl_idname, size_panel, get_prompt)
        yield from create_panel(space_type, 'UI', DreamTexturePanel.bl_idname, inpaint_init_image_panels, get_prompt)
        yield create_panel(space_type, 'UI', DreamTexturePanel.bl_idname, advanced_panel, get_prompt)
        yield create_panel(space_type, 'UI', DreamTexturePanel.bl_idname, actions_panel, get_prompt)

def create_panel(space_type, region_type, parent_id, ctor, get_prompt, use_property_decorate=False):
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
    
    return ctor(SubPanel, space_type, get_prompt)

def prompt_panel(sub_panel, space_type, get_prompt):
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

            structure = next(x for x in prompt_structures if x.id == get_prompt(context).prompt_structure)
            for segment in structure.structure:
                segment_row = layout.row()
                enum_prop = 'prompt_structure_token_' + segment.id + '_enum'
                is_custom = getattr(get_prompt(context), enum_prop) == 'custom'
                if is_custom:
                    segment_row.prop(get_prompt(context), 'prompt_structure_token_' + segment.id)
                enum_cases = DreamPrompt.__annotations__[enum_prop].keywords['items']
                if len(enum_cases) != 1 or enum_cases[0][0] != 'custom':
                    segment_row.prop(get_prompt(context), enum_prop, icon_only=is_custom)
            if get_prompt(context).prompt_structure == file_batch_structure.id:
                layout.template_ID(context.scene, "dream_textures_prompt_file", open="text.open")
            else:
                layout.prop(get_prompt(context), "seamless")
    yield PromptPanel

    class NegativePromptPanel(sub_panel):
        """Create a subpanel for negative prompt input"""
        bl_idname = f"DREAM_PT_dream_panel_negative_prompt_{space_type}"
        bl_label = "Negative"
        bl_parent_id = PromptPanel.bl_idname

        @classmethod
        def poll(self, context):
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

        def draw(self, context):
            super().draw(context)
            layout = self.layout
            layout.use_property_split = True

            layout.prop(get_prompt(context), "width")
            layout.prop(get_prompt(context), "height")
    return SizePanel

def inpaint_init_image_panels(sub_panel, space_type, get_prompt):
    class InpaintPanel(sub_panel):
        """Create a subpanel for inpainting options"""
        bl_idname = f"DREAM_PT_dream_panel_inpaint_{space_type}"
        bl_label = "Inpaint Open Image"
        bl_options = {'DEFAULT_CLOSED'}

        @classmethod
        def poll(self, context):
            if not get_prompt(context).use_init_img:
                for area in context.screen.areas:
                    if area.type == 'IMAGE_EDITOR':
                        if area.spaces.active.image is not None:
                            return True
            return False

        def draw_header(self, context):
            self.layout.prop(get_prompt(context), "use_inpainting", text="")

        def draw(self, context):
            super().draw(context)
            layout = self.layout
            layout.use_property_split = True
            layout.enabled = get_prompt(context).use_inpainting

            layout.label(text="1. Open an Image Editor space")
            layout.label(text="2. Select the 'Paint' editing context")
            layout.label(text="3. Choose the 'Mark Inpaint Area' brush")
            layout.label(text="4. Draw over the area to inpaint")
    yield InpaintPanel

    class InitImagePanel(sub_panel):
        """Create a subpanel for init image options"""
        bl_idname = f"DREAM_PT_dream_panel_init_image_{space_type}"
        bl_label = "Init Image"
        bl_options = {'DEFAULT_CLOSED'}

        @classmethod
        def poll(self, context):
            if get_prompt(context).use_inpainting and InpaintPanel.poll(context):
                return False
            return True

        def draw_header(self, context):
            self.layout.prop(get_prompt(context), "use_init_img", text="")

        def draw(self, context):
            super().draw(context)
            layout = self.layout
            layout.use_property_split = True
            
            layout.enabled = get_prompt(context).use_init_img
            layout.template_ID(context.scene, "init_img", open="image.open")
            layout.prop(get_prompt(context), "strength")
            layout.prop(get_prompt(context), "fit")
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
            
            layout.prop(get_prompt(context), "precision")
            layout.prop(get_prompt(context), "random_seed")
            if not get_prompt(context).random_seed:
                layout.prop(get_prompt(context), "seed")
            # advanced_box.prop(self, "iterations") # Disabled until supported by the addon.
            layout.prop(get_prompt(context), "steps")
            layout.prop(get_prompt(context), "cfg_scale")
            layout.prop(get_prompt(context), "sampler_name")
            layout.prop(get_prompt(context), "show_steps")
    return AdvancedPanel

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

            iterations_row = layout.row()
            iterations_row.enabled = get_prompt(context).prompt_structure != file_batch_structure.id
            iterations_row.prop(get_prompt(context), "iterations")
            
            row = layout.row()
            row.scale_y = 1.5
            if context.scene.dream_textures_progress <= 0:
                if context.scene.dream_textures_info != "":
                    row.label(text=context.scene.dream_textures_info, icon="INFO")
                else:
                    row.operator(DreamTexture.bl_idname, icon="PLAY", text="Generate")
            else:
                disabled_row = row.row()
                disabled_row.prop(context.scene, 'dream_textures_progress', slider=True)
                disabled_row.enabled = False
            if CancelGenerator.poll(context):
                row.operator(CancelGenerator.bl_idname, icon="CANCEL", text="")
            row.operator(ReleaseGenerator.bl_idname, icon="X", text="")
    return ActionsPanel
