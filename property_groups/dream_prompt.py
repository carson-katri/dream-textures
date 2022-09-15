import bpy
from bpy.props import FloatProperty, IntProperty, EnumProperty, BoolProperty, StringProperty, PointerProperty
from ..prompt_engineering import *

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

attributes = {
    # Prompt
    "prompt_structure": EnumProperty(name="Preset", items=prompt_structures_items, description="Fill in a few simple options to create interesting images quickly"),

    # Size
    "width": IntProperty(name="Width", default=512),
    "height": IntProperty(name="Height", default=512),

    # Simple Options
    "seamless": BoolProperty(name="Seamless", default=False, description="Enables seamless/tilable image generation"),

    # Advanced
    "show_advanced": BoolProperty(name="", default=False),
    "seed": IntProperty(name="Seed", default=-1, description="Seed for RNG. Using the same seed will give the same image. A seed of '-1' will pick a random seed each time"),
    "full_precision": BoolProperty(name="Full Precision", default=False, description="Whether to use full precision or half precision floats. Full precision is slower, but required by some GPUs"),
    "iterations": IntProperty(name="Iterations", default=1, min=1, description="How many images to generate"),
    "steps": IntProperty(name="Steps", default=25, min=1),
    "cfgscale": FloatProperty(name="CFG Scale", default=7.5, min=1, description="How strongly the prompt influences the image"),
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

DreamPrompt.generate_prompt = generate_prompt
DreamPrompt.get_prompt_subject = get_prompt_subject