import bpy
from bpy.props import FloatProperty, IntProperty, EnumProperty, BoolProperty, StringProperty, IntVectorProperty
import os
import sys
from typing import _AnnotatedAlias

from ..generator_process.actions.detect_seamless import SeamlessAxes
from ..generator_process.actions.prompt_to_image import Optimizations, Scheduler, StepPreviewMode, Pipeline
from ..generator_process.actions.huggingface_hub import ModelType
from ..prompt_engineering import *
from ..preferences import StableDiffusionPreferences
from .dream_prompt_validation import validate

from functools import reduce

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
    SeamlessAxes.AUTO.bpy_enum('Detect from source image when modifying or inpainting, off otherwise', -1),
    SeamlessAxes.OFF.bpy_enum('', 0),
    SeamlessAxes.HORIZONTAL.bpy_enum('', 1),
    SeamlessAxes.VERTICAL.bpy_enum('', 2),
    SeamlessAxes.BOTH.bpy_enum('', 3),
]

def modify_action_source_type(self, context):
    return [
        ('color', 'Color', 'Use the color information from the image', 1),
        ('depth_generated', 'Color and Generated Depth', 'Use MiDaS to infer the depth of the initial image and include it in the conditioning. Can give results that more closely match the composition of the source image', 2),
        ('depth_map', 'Color and Depth Map', 'Specify a secondary image to use as the depth map. Can give results that closely match the composition of the depth map', 3),
        ('depth', 'Depth', 'Treat the initial image as a depth map, and ignore any color. Matches the composition of the source image without any color influence', 4),
    ]

def model_options(self, context):
    match Pipeline[self.pipeline]:
        case Pipeline.STABLE_DIFFUSION:
            def model_case(model, i):
                return (
                    model.model_base,
                    model.model_base.replace('models--', '').replace('--', '/'),
                    ModelType[model.model_type].name,
                    i
                )
            models = {}
            for i, model in enumerate(context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences.installed_models):
                if model.model_type not in models:
                    models[model.model_type] = [model_case(model, i)]
                else:
                    models[model.model_type].append(model_case(model, i))
            return reduce(
                lambda a, b: a + [None] + sorted(b, key=lambda m: m[0]),
                [
                    models[group]
                    for group in sorted(models.keys())
                ],
                []
            )
        case Pipeline.STABILITY_SDK:
            return [
                ("stable-diffusion-v1", "Stable Diffusion v1.4", ModelType.PROMPT_TO_IMAGE.name),
                ("stable-diffusion-v1-5", "Stable Diffusion v1.5", ModelType.PROMPT_TO_IMAGE.name),
                ("stable-diffusion-512-v2-0", "Stable Diffusion v2.0", ModelType.PROMPT_TO_IMAGE.name),
                ("stable-diffusion-768-v2-0", "Stable Diffusion v2.0-768", ModelType.PROMPT_TO_IMAGE.name),
                ("stable-diffusion-512-v2-1", "Stable Diffusion v2.1", ModelType.PROMPT_TO_IMAGE.name),
                ("stable-diffusion-768-v2-1", "Stable Diffusion v2.1-768", ModelType.PROMPT_TO_IMAGE.name),
                None,
                ("stable-inpainting-v1-0", "Stable Inpainting v1.0", ModelType.INPAINTING.name),
                ("stable-inpainting-512-v2-0", "Stable Inpainting v2.0", ModelType.INPAINTING.name),
            ]

def pipeline_options(self, context):
    return [
        (Pipeline.STABLE_DIFFUSION.name, 'Stable Diffusion', 'Stable Diffusion on your own hardware', 1),
        (Pipeline.STABILITY_SDK.name, 'DreamStudio', 'Cloud compute via DreamStudio', 2)
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
    "pipeline": EnumProperty(name="Pipeline", items=pipeline_options, default=1 if Pipeline.local_available() else 2, description="Specify which model and target should be used."),
    "model": EnumProperty(name="Model", items=model_options, description="Specify which model to use for inference"),

    # Prompt
    "prompt_structure": EnumProperty(name="Preset", items=prompt_structures_items, description="Fill in a few simple options to create interesting images quickly"),
    "use_negative_prompt": BoolProperty(name="Use Negative Prompt", default=False),
    "negative_prompt": StringProperty(name="Negative Prompt", description="The model will avoid aspects of the negative prompt"),

    # Size
    "use_size": BoolProperty(name="Manual Size", default=False),
    "width": IntProperty(name="Width", default=512, min=64, step=64),
    "height": IntProperty(name="Height", default=512, min=64, step=64),

    # Simple Options
    "seamless_axes": EnumProperty(name="Seamless Axes", items=seamless_axes, default=SeamlessAxes.AUTO.id, description="Specify which axes should be seamless/tilable"),

    # Advanced
    "show_advanced": BoolProperty(name="", default=False),
    "random_seed": BoolProperty(name="Random Seed", default=True, description="Randomly pick a seed"),
    "seed": StringProperty(name="Seed", default="0", description="Manually pick a seed", update=seed_clamp),
    "iterations": IntProperty(name="Iterations", default=1, min=1, description="How many images to generate"),
    "steps": IntProperty(name="Steps", default=25, min=1),
    "cfg_scale": FloatProperty(name="CFG Scale", default=7.5, min=1, soft_min=1.01, description="How strongly the prompt influences the image"),
    "scheduler": EnumProperty(name="Scheduler", items=scheduler_options, default=3), # defaults to "DPM Solver Multistep"
    "step_preview_mode": EnumProperty(name="Step Preview", description="Displays intermediate steps in the Image Viewer. Disabling can speed up generation", items=step_preview_mode_options, default=1),

    # Init Image
    "use_init_img": BoolProperty(name="Use Init Image", default=False),
    "init_img_src": EnumProperty(name=" ", items=init_image_sources, default="file"),
    "init_img_action": EnumProperty(name="Action", items=init_image_actions_filtered, default=1),
    "strength": FloatProperty(name="Noise Strength", description="The ratio of noise:image. A higher value gives more 'creative' results", default=0.75, min=0, max=1, soft_min=0.01, soft_max=0.99),
    "fit": BoolProperty(name="Fit to width/height", description="Resize the source image to match the specified size", default=True),
    "use_init_img_color": BoolProperty(name="Color Correct", default=True),
    "modify_action_source_type": EnumProperty(name="Image Type", items=modify_action_source_type, default=1, description="What kind of data is the source image"),
    
    # Inpaint
    "inpaint_mask_src": EnumProperty(name="Mask Source", items=inpaint_mask_sources_filtered, default=1),
    "inpaint_replace": FloatProperty(name="Replace", description="Replaces the masked area with a specified amount of noise, can create more extreme changes. Values of 0 or 1 will give the best results", min=0, max=1, default=0),
    "text_mask": StringProperty(name="Mask Prompt"),
    "text_mask_confidence": FloatProperty(name="Confidence Threshold", description="How confident the segmentation model needs to be", default=0.5, min=0),

    # Outpaint
    "outpaint_origin": IntVectorProperty(name="Origin", default=(0, 448), size=2, description="The position of the outpaint area relative to the top left corner of the image. A value of (0, 512) will outpaint from the bottom of a 512x512 image"),

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
    attributes[f"optimizations_{optim}"] = BoolProperty(name=optim.replace('_', ' ').title(), default=default)
attributes["optimizations_attention_slice_size_src"] = EnumProperty(name="Attention Slice Size", items=(
    ("auto", "Automatic", "", 1),
    ("manual", "Manual", "", 2),
), default=1)
attributes["optimizations_attention_slice_size"] = IntProperty(name="Attention Slice Size", default=1, min=1)
attributes["optimizations_batch_size"] = IntProperty(name="Batch Size", default=1, min=1)

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
    if not args['use_negative_prompt']:
        args['negative_prompt'] = None
    args['prompt'] = self.generate_prompt()
    args['seed'] = self.get_seed()
    args['optimizations'] = self.get_optimizations()
    args['scheduler'] = Scheduler(args['scheduler'])
    args['step_preview_mode'] = StepPreviewMode(args['step_preview_mode'])
    args['pipeline'] = Pipeline[args['pipeline']]
    args['outpaint_origin'] = (args['outpaint_origin'][0], args['outpaint_origin'][1])
    args['key'] = bpy.context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences.dream_studio_key
    args['seamless_axes'] = SeamlessAxes(args['seamless_axes'])
    args['width'] = args['width'] if args['use_size'] else None
    args['height'] = args['height'] if args['use_size'] else None
    return args

DreamPrompt.generate_prompt = generate_prompt
DreamPrompt.get_prompt_subject = get_prompt_subject
DreamPrompt.get_seed = get_seed
DreamPrompt.get_optimizations = get_optimizations
DreamPrompt.generate_args = generate_args
DreamPrompt.validate = validate