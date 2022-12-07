import bpy
from bpy.props import FloatProperty, IntProperty, EnumProperty, BoolProperty, StringProperty
import os
import sys
from typing import _AnnotatedAlias
from ..absolute_path import absolute_path
from ..generator_process.actions.prompt_to_image import Optimizations, Scheduler, StepPreviewMode, Pipeline
from ..generator_process import Generator
from ..prompt_engineering import *

scheduler_options = [(scheduler.value, scheduler.value, '') for scheduler in Scheduler]

step_preview_mode_options = [(mode.value, mode.value, '') for mode in StepPreviewMode]

precision_options = [
    ('auto', 'Automatic', "", 1),
    ('float32', 'Full Precision (float32)', "", 2),
    ('autocast', 'Autocast', "", 3),
    ('float16', 'Half Precision (float16)', "", 4),
]

init_image_sources = [
    ('file', 'File', '', 'IMAGE_DATA', 1),
    ('open_editor', 'Open Image', '', 'TPAINT_HLT', 2),
]

init_image_actions = [
    ('modify', 'Modify', 'Combine the initial image with noise to influence the output', 'IMAGE', 1),
    ('inpaint', 'Inpaint', 'Fill in any masked areas', 'TPAINT_HLT', 2),
    ('outpaint', 'Outpaint', 'Extend the image in a specific direction', 'FULLSCREEN_ENTER', 3),
]

def init_image_actions_filtered(self, context):
    available = Pipeline[self.pipeline].init_img_actions()
    return list(filter(lambda x: x[0] in available, init_image_actions))

inpaint_mask_sources = [
    ('alpha', 'Alpha Channel', '', 1),
    ('prompt', 'Prompt', '', 2),
]

def inpaint_mask_sources_filtered(self, context):
    available = Pipeline[self.pipeline].inpaint_mask_sources()
    return list(filter(lambda x: x[0] in available, inpaint_mask_sources))

seamless_axes = [
    ('x', 'X', '', 1),
    ('y', 'Y', '', 2),
    ('xy', 'Both', '', 3),
]

def model_options(self, context):
    match Pipeline[self.pipeline]:
        case Pipeline.STABLE_DIFFUSION:
            return [(m.model, os.path.basename(m.model).replace('models--', '').replace('--', '/'), '', i) for i, m in enumerate(context.preferences.addons[__package__.split('.')[0]].preferences.installed_models)]
        case Pipeline.STABILITY_SDK:
            return [(x, x, '') for x in [
                "stable-diffusion-v1",
                "stable-diffusion-v1-5",
                "stable-diffusion-512-v2-0",
                "stable-diffusion-768-v2-0",
                "stable-inpainting-v1-0",
                "stable-inpainting-512-v2-0"
            ]]

def pipeline_options(self, context):
    def options():
        if Pipeline.local_available():
            yield (Pipeline.STABLE_DIFFUSION.name, 'Stable Diffusion', 'Stable Diffusion on your own hardware', 1)
        if len(context.preferences.addons[__package__.split('.')[0]].preferences.dream_studio_key) > 0:
            yield (Pipeline.STABILITY_SDK.name, 'DreamStudio', 'Cloud compute via DreamStudio', 2)
    return [*options()]

def seed_clamp(self, ctx):
    # clamp seed right after input to make it clear what the limits are
    try:
        s = str(max(0,min(int(float(self.seed)),2**32-1))) # float() first to make sure any seed that is a number gets clamped, not just ints
        if s != self.seed:
            self.seed = s
    except (ValueError, OverflowError):
        pass # will get hashed once generated

attributes = {
    "pipeline": EnumProperty(name="Pipeline", items=pipeline_options, default=1 if Pipeline.local_available() else 2, description="Specify which model and target should be used."),
    "model": EnumProperty(name="Model", items=model_options, description="Specify which model to use for inference"),

    # Prompt
    "prompt_structure": EnumProperty(name="Preset", items=prompt_structures_items, description="Fill in a few simple options to create interesting images quickly"),
    "use_negative_prompt": BoolProperty(name="Use Negative Prompt", default=False),
    "negative_prompt": StringProperty(name="Negative Prompt", description="The model will avoid aspects of the negative prompt"),

    # Size
    "width": IntProperty(name="Width", default=512, min=64, step=64),
    "height": IntProperty(name="Height", default=512, min=64, step=64),

    # Simple Options
    "seamless": BoolProperty(name="Seamless", default=False, description="Enables seamless/tilable image generation"),
    "seamless_axes": EnumProperty(name="Seamless Axes", items=seamless_axes, default='xy', description="Specify which axes should be seamless/tilable"),

    # Advanced
    "show_advanced": BoolProperty(name="", default=False),
    "random_seed": BoolProperty(name="Random Seed", default=True, description="Randomly pick a seed"),
    "seed": StringProperty(name="Seed", default="0", description="Manually pick a seed", update=seed_clamp),
    "iterations": IntProperty(name="Iterations", default=1, min=1, description="How many images to generate"),
    "steps": IntProperty(name="Steps", default=25, min=1),
    "cfg_scale": FloatProperty(name="CFG Scale", default=7.5, min=1, soft_min=1.01, description="How strongly the prompt influences the image"),
    "scheduler": EnumProperty(name="Scheduler", items=scheduler_options, default=0),
    "step_preview_mode": EnumProperty(name="Step Preview", description="Displays intermediate steps in the Image Viewer. Disabling can speed up generation", items=step_preview_mode_options, default=1),

    # Init Image
    "use_init_img": BoolProperty(name="Use Init Image", default=False),
    "init_img_src": EnumProperty(name=" ", items=init_image_sources, default="file"),
    "init_img_action": EnumProperty(name="Action", items=init_image_actions_filtered, default=1),
    "strength": FloatProperty(name="Noise Strength", description="The ratio of noise:image. A higher value gives more 'creative' results", default=0.75, min=0, max=1, soft_min=0.01, soft_max=0.99),
    "fit": BoolProperty(name="Fit to width/height", default=True),
    "use_init_img_color": BoolProperty(name="Color Correct", default=True),
    
    # Inpaint
    "inpaint_mask_src": EnumProperty(name="Mask Source", items=inpaint_mask_sources_filtered, default=1),
    "inpaint_replace": FloatProperty(name="Replace", description="Replaces the masked area with a specified amount of noise, can create more extreme changes. Values of 0 or 1 will give the best results", min=0, max=1, default=0),
    "text_mask": StringProperty(name="Mask Prompt"),
    "text_mask_confidence": FloatProperty(name="Confidence Threshold", description="How confident the segmentation model needs to be", default=0.5, min=0),

    # Outpaint
    "outpaint_top": IntProperty(name="Top", default=64, step=64, min=0),
    "outpaint_right": IntProperty(name="Right", default=64, step=64, min=0),
    "outpaint_bottom": IntProperty(name="Bottom", default=64, step=64, min=0),
    "outpaint_left": IntProperty(name="Left", default=64, step=64, min=0),
    "outpaint_blend": IntProperty(name="Blend", description="Gaussian blur amount to apply to the extended area", default=16, min=0),

    # Resulting image
    "hash": StringProperty(name="Image Hash"),
}

default_optimizations = Optimizations()
for optim in dir(Optimizations):
    if optim.startswith('_'):
        continue
    if hasattr(Optimizations.__annotations__, optim):
        annotation = Optimizations.__annotations__[optim]
        if annotation != bool or (annotation is _AnnotatedAlias and annotation.__origin__ != bool):
            continue
    default = getattr(default_optimizations, optim, None)
    if default is not None and not isinstance(getattr(default_optimizations, optim), bool):
        continue
    setattr(default_optimizations, optim, True)
    if default_optimizations.can_use(optim, "mps" if sys.platform == "darwin" else "cuda"):
        attributes[f"optimizations_{optim}"] = BoolProperty(name=optim.replace('_', ' ').title(), default=default)
attributes["optimizations_attention_slice_size_src"] = EnumProperty(name="Attention Slice Size", items=(
    ("auto", "Automatic", "", 1),
    ("manual", "Manual", "", 2),
), default=1)
attributes["optimizations_attention_slice_size"] = IntProperty(name="Attention Slice Size", default=1)

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
    return structure.generate(dotdict(tokens)) + (f" [{self.negative_prompt}]" if self.use_negative_prompt else "")

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

def get_optimizations(self: DreamPrompt):
    optimizations = Optimizations()
    for prop in dir(self):
        split_name = prop.replace('optimizations_', '')
        if prop.startswith('optimizations_') and hasattr(optimizations, split_name):
            setattr(optimizations, split_name, getattr(self, prop))
    if self.optimizations_attention_slice_size_src == 'auto':
        optimizations.attention_slice_size = 'auto'
    return optimizations

def generate_args(self):
    args = { key: getattr(self, key) for key in DreamPrompt.__annotations__ }
    args['prompt'] = self.generate_prompt()
    args['seed'] = self.get_seed()
    args['optimizations'] = self.get_optimizations()
    args['scheduler'] = Scheduler(args['scheduler'])
    args['step_preview_mode'] = StepPreviewMode(args['step_preview_mode'])
    args['pipeline'] = Pipeline[args['pipeline']]
    args['key'] = bpy.context.preferences.addons[__package__.split('.')[0]].preferences.dream_studio_key
    return args

DreamPrompt.generate_prompt = generate_prompt
DreamPrompt.get_prompt_subject = get_prompt_subject
DreamPrompt.get_seed = get_seed
DreamPrompt.get_optimizations = get_optimizations
DreamPrompt.generate_args = generate_args
