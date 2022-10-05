import enum
from bpy.types import Panel
from ...pil_to_image import *
from ...prompt_engineering import *
from ...operators.dream_texture import DreamTexture, ReleaseGenerator
from ...operators.open_latest_version import OpenLatestVersion, is_force_show_download, new_version_available
from ...operators.view_history import ImportPromptFile
from ..space_types import SPACE_TYPES
from ...property_groups.dream_prompt import DreamPrompt

def dream_texture_panels():
    for space_type in SPACE_TYPES:
        class BasePanel:
            bl_category = "Dream"
            bl_space_type = space_type
            bl_region_type = 'UI'
        class SubPanel(BasePanel):
            bl_category = "Dream"
            bl_space_type = space_type
            bl_region_type = 'UI'
            bl_parent_id = f"DREAM_PT_dream_panel_{space_type}"

        class DreamTexturePanel(Panel, BasePanel):
            """Creates a Panel in the scene context of the properties editor"""
            bl_label = "Dream Texture"
            bl_idname = f"DREAM_PT_dream_panel_{space_type}"

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
                layout.use_property_decorate = False

                if is_force_show_download():
                    layout.operator(OpenLatestVersion.bl_idname, icon="IMPORT", text="Download Latest Release")
                elif new_version_available():
                    layout.operator(OpenLatestVersion.bl_idname, icon="IMPORT")

        DreamTexturePanel.__name__ = f"DREAM_PT_dream_panel_{space_type}"
        yield DreamTexturePanel
    
        class PromptPanel(Panel, SubPanel):
            """Create a subpanel for prompt input"""
            bl_idname = f"DREAM_PT_dream_panel_prompt_{space_type}"
            bl_label = "Prompt"
            bl_parent_id = f"DREAM_PT_dream_panel_{space_type}"

            def draw_header_preset(self, context):
                layout = self.layout
                layout.prop(context.scene.dream_textures_prompt, "prompt_structure", text="")

            def draw(self, context):
                layout = self.layout
                layout.use_property_split = True
                layout.use_property_decorate = False
                scene = context.scene

                structure = next(x for x in prompt_structures if x.id == scene.dream_textures_prompt.prompt_structure)
                for segment in structure.structure:
                    segment_row = layout.row()
                    enum_prop = 'prompt_structure_token_' + segment.id + '_enum'
                    is_custom = getattr(scene.dream_textures_prompt, enum_prop) == 'custom'
                    if is_custom:
                        segment_row.prop(scene.dream_textures_prompt, 'prompt_structure_token_' + segment.id)
                    enum_cases = DreamPrompt.__annotations__[enum_prop].keywords['items']
                    if len(enum_cases) != 1 or enum_cases[0][0] != 'custom':
                        segment_row.prop(scene.dream_textures_prompt, enum_prop, icon_only=is_custom)
                layout.prop(scene.dream_textures_prompt, "seamless")
        yield PromptPanel
    
        class NegativePromptPanel(Panel, SubPanel):
            """Create a subpanel for negative prompt input"""
            bl_idname = f"DREAM_PT_dream_panel_negative_prompt_{space_type}"
            bl_label = "Negative"
            bl_parent_id = PromptPanel.bl_idname

            def draw_header(self, context):
                layout = self.layout
                layout.prop(context.scene.dream_textures_prompt, "use_negative_prompt", text="")

            def draw(self, context):
                layout = self.layout
                layout.use_property_split = True
                layout.use_property_decorate = False
                layout.enabled = context.scene.dream_textures_prompt.use_negative_prompt
                scene = context.scene

                layout.prop(scene.dream_textures_prompt, "negative_prompt")
        yield NegativePromptPanel
    
        class SizePanel(Panel, SubPanel):
            """Create a subpanel for size options"""
            bl_idname = f"DREAM_PT_dream_panel_size_{space_type}"
            bl_label = "Size"

            def draw(self, context):
                layout = self.layout
                layout.use_property_split = True
                layout.use_property_decorate = False
                scene = context.scene

                layout.prop(scene.dream_textures_prompt, "width")
                layout.prop(scene.dream_textures_prompt, "height")
        yield SizePanel
    
        class InpaintPanel(Panel, SubPanel):
            """Create a subpanel for inpainting options"""
            bl_idname = f"DREAM_PT_dream_panel_inpaint_{space_type}"
            bl_label = "Inpaint Open Image"
            bl_options = {'DEFAULT_CLOSED'}

            @classmethod
            def poll(self, context):
                if not context.scene.dream_textures_prompt.use_init_img:
                    for area in context.screen.areas:
                        if area.type == 'IMAGE_EDITOR':
                            if area.spaces.active.image is not None:
                                return True
                return False

            def draw_header(self, context):
                self.layout.prop(context.scene.dream_textures_prompt, "use_inpainting", text="")

            def draw(self, context):
                layout = self.layout
                layout.use_property_split = True
                layout.use_property_decorate = False
                layout.enabled = context.scene.dream_textures_prompt.use_inpainting

                layout.label(text="1. Open an Image Editor space")
                layout.label(text="2. Select the 'Paint' editing context")
                layout.label(text="3. Choose the 'Mark Inpaint Area' brush")
                layout.label(text="4. Draw over the area to inpaint")
        yield InpaintPanel
    
        class InitImagePanel(Panel, SubPanel):
            """Create a subpanel for init image options"""
            bl_idname = f"DREAM_PT_dream_panel_init_image_{space_type}"
            bl_label = "Init Image"
            bl_options = {'DEFAULT_CLOSED'}

            @classmethod
            def poll(self, context):
                if context.scene.dream_textures_prompt.use_inpainting and InpaintPanel.poll(context):
                    return False
                return True

            def draw_header(self, context):
                self.layout.prop(context.scene.dream_textures_prompt, "use_init_img", text="")

            def draw(self, context):
                layout = self.layout
                layout.use_property_split = True
                layout.use_property_decorate = False
                scene = context.scene
                
                layout.enabled = context.scene.dream_textures_prompt.use_init_img
                layout.template_ID(context.scene, "init_img", open="image.open")
                layout.prop(scene.dream_textures_prompt, "strength")
                layout.prop(scene.dream_textures_prompt, "fit")
        yield InitImagePanel
    
        class AdvancedPanel(Panel, SubPanel):
            """Create a subpanel for advanced options"""
            bl_idname = f"DREAM_PT_dream_panel_advanced_{space_type}"
            bl_label = "Advanced"
            bl_options = {'DEFAULT_CLOSED'}

            def draw(self, context):
                layout = self.layout
                layout.use_property_split = True
                layout.use_property_decorate = False
                scene = context.scene
                
                layout.prop(scene.dream_textures_prompt, "precision")
                layout.prop(scene.dream_textures_prompt, "random_seed")
                if not scene.dream_textures_prompt.random_seed:
                    layout.prop(scene.dream_textures_prompt, "seed")
                # advanced_box.prop(self, "iterations") # Disabled until supported by the addon.
                layout.prop(scene.dream_textures_prompt, "steps")
                layout.prop(scene.dream_textures_prompt, "cfg_scale")
                layout.prop(scene.dream_textures_prompt, "sampler_name")
                layout.prop(scene.dream_textures_prompt, "show_steps")
        yield AdvancedPanel
    
        class ActionsPanel(Panel, SubPanel):
            """Create a subpanel for actions"""
            bl_idname = f"DREAM_PT_dream_panel_actions_{space_type}"
            bl_label = "Advanced"
            bl_options = {'HIDE_HEADER'}

            def draw(self, context):
                layout = self.layout
                layout.use_property_split = True
                layout.use_property_decorate = False
                
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
                row.operator(ReleaseGenerator.bl_idname, icon="X", text="")
        yield ActionsPanel
