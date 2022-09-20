import bpy
from bpy.types import Panel
from ..async_loop import *
from ..pil_to_image import *
from ..prompt_engineering import *
from ..operators.dream_texture import image_has_alpha
from ..operators.view_history import ViewHistory
from ..operators.open_latest_version import OpenLatestVersion, new_version_available
from ..operators.help_panel import HelpPanel

def draw_panel(self, context):
    layout = self.layout
    scene = context.scene

    if new_version_available():
        layout.operator(OpenLatestVersion.bl_idname, icon="IMPORT")

    prompt_box = layout.box()
    prompt_box_heading = prompt_box.row()
    prompt_box_heading.label(text="Prompt")
    prompt_box_heading.prop(scene.dream_textures_prompt, "prompt_structure")
    structure = next(x for x in prompt_structures if x.id == scene.dream_textures_prompt.prompt_structure)
    for segment in structure.structure:
        segment_row = prompt_box.row()
        enum_prop = 'prompt_structure_token_' + segment.id + '_enum'
        is_custom = getattr(scene.dream_textures_prompt, enum_prop) == 'custom'
        if is_custom:
            segment_row.prop(scene.dream_textures_prompt, 'prompt_structure_token_' + segment.id)
        segment_row.prop(scene.dream_textures_prompt, enum_prop, icon_only=is_custom)
    
    size_box = layout.box()
    size_box.label(text="Configuration")
    size_box.prop(scene.dream_textures_prompt, "width")
    size_box.prop(scene.dream_textures_prompt, "height")
    size_box.prop(scene.dream_textures_prompt, "seamless")
    
    for area in context.screen.areas:
        if area.type == 'IMAGE_EDITOR':
            if area.spaces.active.image is not None and image_has_alpha(area.spaces.active.image):
                inpainting_box = layout.box()
                inpainting_heading = inpainting_box.row()
                inpainting_heading.prop(scene.dream_textures_prompt, "use_inpainting")
                inpainting_heading.label(text="Inpaint Open Image")
                break

    if not scene.dream_textures_prompt.use_inpainting:
        init_img_box = layout.box()
        init_img_heading = init_img_box.row()
        init_img_heading.prop(scene.dream_textures_prompt, "use_init_img")
        init_img_heading.label(text="Init Image")
        if scene.dream_textures_prompt.use_init_img:
            init_img_box.template_ID(context.scene, "init_img", open="image.open")
            init_img_box.prop(scene.dream_textures_prompt, "strength")
            init_img_box.prop(scene.dream_textures_prompt, "fit")

    advanced_box = layout.box()
    advanced_box_heading = advanced_box.row()
    advanced_box_heading.prop(scene.dream_textures_prompt, "show_advanced", icon="DOWNARROW_HLT" if scene.dream_textures_prompt.show_advanced else "RIGHTARROW_THIN", emboss=False, icon_only=True)
    advanced_box_heading.label(text="Advanced Configuration")
    if scene.dream_textures_prompt.show_advanced:
        advanced_box.prop(scene.dream_textures_prompt, "full_precision")
        advanced_box.prop(scene.dream_textures_prompt, "seed")
        # advanced_box.prop(self, "iterations") # Disabled until supported by the addon.
        advanced_box.prop(scene.dream_textures_prompt, "steps")
        advanced_box.prop(scene.dream_textures_prompt, "cfgscale")
        advanced_box.prop(scene.dream_textures_prompt, "sampler")
        advanced_box.prop(scene.dream_textures_prompt, "show_steps")
    
    row = layout.row()
    row.operator(ViewHistory.bl_idname, icon="RECOVER_LAST")
    row.operator(HelpPanel.bl_idname, icon="QUESTION", text="")
    row = layout.row()
    row.scale_y = 1.5
    row.operator("shade.dream_texture", icon="PLAY", text="Generate")

class DREAM_PT_dream_panel(Panel):
    """Creates a Panel in the scene context of the properties editor"""
    bl_label = "Dream Texture"
    bl_category = "Dream"
    bl_idname = "DREAM_PT_dream_panel"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'

    def draw(self, context):
        draw_panel(self, context)

class DREAM_PT_dream_node_panel(Panel):
    bl_idname = "DREAM_PT_dream_node_panel"
    bl_label = "Dream Texture"
    bl_category = "Dream"
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'

    def draw(self, context):
        draw_panel(self,context)

classes = (
    DREAM_PT_dream_panel,
    DREAM_PT_dream_node_panel,
)

def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)

def unregister():
    from bpy.utils import unregister_class
    for cls in classes:
        unregister_class(cls)
