import bpy
from bpy.props import FloatProperty, IntProperty, EnumProperty, BoolProperty, StringProperty, PointerProperty
from ..prompt_engineering import *
import sys

sampler_options = [
    ("ddim", "DDIM", "", 1),
    ("plms", "PLMS", "", 2),
    ("k_lms", "KLMS", "", 3),
    ("k_dpm_2", "KDPM_2", "", 4),
    ("k_dpm_2_a", "KDPM_2A", "", 5),
    ("k_euler", "KEULER", "", 6),
    ("k_euler_a", "KEULER_A", "", 7),
    ("k_heun", "KHEUN", "", 8),
]

def seed_clamp(self, ctx):
    # clamp seed right after input to make it clear what the limits are
    try:
        s = str(max(0,min(int(float(self.seed)),2**32-1))) # float() first to make sure any seed that is a number gets clamped, not just ints
        if s != self.seed:
            self.seed = s
    except (ValueError, OverflowError):
        pass # will get hashed once generated

attributes = {
    # Prompt
    "prompt_structure": EnumProperty(name="Preset", items=prompt_structures_items, description="Fill in a few simple options to create interesting images quickly"),

    # Size
    "width": IntProperty(name="Width", default=512, min=64, step=64),
    "height": IntProperty(name="Height", default=512, min=64, step=64),

    # Simple Options
    "seamless": BoolProperty(name="Seamless", default=False, description="Enables seamless/tilable image generation"),

    # Advanced
    "show_advanced": BoolProperty(name="", default=False),
    "random_seed": BoolProperty(name="Random Seed", default=True, description="Randomly pick a seed"),
    "seed": StringProperty(name="Seed", default="0", description="Manually pick a seed", update=seed_clamp),
    "full_precision": BoolProperty(name="Full Precision", default=True if sys.platform == 'darwin' else False, description="Whether to use full precision or half precision floats. Full precision is slower, but required by some GPUs"),
    "iterations": IntProperty(name="Iterations", default=1, min=1, description="How many images to generate"),
    "steps": IntProperty(name="Steps", default=25, min=1),
    "cfg_scale": FloatProperty(name="CFG Scale", default=7.5, min=1, description="How strongly the prompt influences the image"),
    "sampler": EnumProperty(name="Sampler", items=sampler_options, default=3),
    "show_steps": BoolProperty(name="Show Steps", description="Displays intermediate steps in the Image Viewer. Disabling can speed up generation", default=True),

    # Init Image
    "use_init_img": BoolProperty(name="", default=False),
    "use_inpainting": BoolProperty(name="", default=False),
    "strength": FloatProperty(name="Strength", default=0.75, min=0, max=1),
    "fit": BoolProperty(name="Fit to width/height", default=True),
}

def map_structure_token_items(value):
    return (value[0], value[1], '')
for structure in prompt_structures:
    for token in structure.structure:
        if not isinstance(token, str):
            attributes['prompt_structure_token_' + token.id] = StringProperty(name=token.label)
            attributes['prompt_structure_token_' + token.id + '_enum'] = EnumProperty(
                name=token.label,
                items=[('custom', 'Custom', '')] + list(map(map_structure_token_items, token.values)),
                default='custom' if len(token.values) == 0 else token.values[0][0],
            )

DreamPrompt = type('DreamPrompt', (bpy.types.PropertyGroup,), {
    "bl_label": "DreamPrompt",
    "bl_idname": "dream_textures.DreamPrompt",
    "__annotations__": attributes,
})

def generate_prompt(self):
    structure = next(x for x in prompt_structures if x.id == self.prompt_structure)
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
    tokens = {}
    for segment in structure.structure:
        enum_value = getattr(self, 'prompt_structure_token_' + segment.id + '_enum')
        if enum_value == 'custom':
            tokens[segment.id] = getattr(self, 'prompt_structure_token_' + segment.id)
        else:
            tokens[segment.id] = next(x for x in segment.values if x[0] == enum_value)[1]
    return structure.generate(dotdict(tokens))

def get_prompt_subject(self):
    structure = next(x for x in prompt_structures if x.id == self.prompt_structure)
    for segment in structure.structure:
        if segment.id == 'subject':
            return getattr(self, 'prompt_structure_token_' + segment.id)
    return self.generate_prompt()

def get_seed(self):
    import numpy
    numpy.random.randn()
    if self.random_seed:
        return None # let stable diffusion automatically pick one
    try:
        return max(0,min(int(float(self.seed)),2**32-1)) # clamp int
    except (ValueError, OverflowError):
        h = hash(self.seed) # not an int, let's hash it!
        if h < 0:
            h = ~h
        return (h & 0xFFFFFFFF) ^ (h >> 32) # 64 bit hash down to 32 bits

DreamPrompt.generate_prompt = generate_prompt
DreamPrompt.get_prompt_subject = get_prompt_subject
DreamPrompt.get_seed = get_seed

def draw_dream_prompt_ui(context, layout, dream_prompt):
    prompt_box = layout.box()
    prompt_box_heading = prompt_box.row()
    prompt_box_heading.label(text="Prompt")
    prompt_box_heading.prop(dream_prompt, "prompt_structure")
    structure = next(x for x in prompt_structures if x.id == dream_prompt.prompt_structure)
    for segment in structure.structure:
        segment_row = prompt_box.row()
        enum_prop = 'prompt_structure_token_' + segment.id + '_enum'
        is_custom = getattr(dream_prompt, enum_prop) == 'custom'
        if is_custom:
            segment_row.prop(dream_prompt, 'prompt_structure_token_' + segment.id)
        segment_row.prop(dream_prompt, enum_prop, icon_only=is_custom)
    
    size_box = layout.box()
    size_box.label(text="Configuration")
    size_box.prop(dream_prompt, "width")
    size_box.prop(dream_prompt, "height")
    size_box.prop(dream_prompt, "seamless")
    
    if not dream_prompt.use_init_img:
        for area in context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                if area.spaces.active.image is not None:
                    inpainting_box = layout.box()
                    inpainting_heading = inpainting_box.row()
                    inpainting_heading.prop(dream_prompt, "use_inpainting")
                    inpainting_heading.label(text="Inpaint Open Image")
                    break

    if not dream_prompt.use_inpainting or area.spaces.active.image is None:
        init_img_box = layout.box()
        init_img_heading = init_img_box.row()
        init_img_heading.prop(dream_prompt, "use_init_img")
        init_img_heading.label(text="Init Image")
        if dream_prompt.use_init_img:
            init_img_box.template_ID(context.scene, "init_img", open="image.open")
            init_img_box.prop(dream_prompt, "strength")
            init_img_box.prop(dream_prompt, "fit")

    advanced_box = layout.box()
    advanced_box_heading = advanced_box.row()
    advanced_box_heading.prop(dream_prompt, "show_advanced", icon="DOWNARROW_HLT" if dream_prompt.show_advanced else "RIGHTARROW_THIN", emboss=False, icon_only=True)
    advanced_box_heading.label(text="Advanced Configuration")
    if dream_prompt.show_advanced:
        if sys.platform not in {'darwin'}:
            advanced_box.prop(dream_prompt, "full_precision")
        advanced_box.prop(dream_prompt, "random_seed")
        seed_row = advanced_box.row()
        seed_row.prop(dream_prompt, "seed")
        seed_row.enabled = not dream_prompt.random_seed
        # advanced_box.prop(self, "iterations") # Disabled until supported by the addon.
        advanced_box.prop(dream_prompt, "steps")
        advanced_box.prop(dream_prompt, "cfg_scale")
        advanced_box.prop(dream_prompt, "sampler")
        advanced_box.prop(dream_prompt, "show_steps")