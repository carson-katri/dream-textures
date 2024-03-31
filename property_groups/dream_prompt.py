import bpy
from bpy.props import FloatProperty, IntProperty, EnumProperty, BoolProperty, StringProperty, IntVectorProperty, CollectionProperty
import os
import sys
from typing import _AnnotatedAlias

from ..generator_process.actions.detect_seamless import SeamlessAxes
from ..generator_process.actions.prompt_to_image import Optimizations, Scheduler, StepPreviewMode
from ..prompt_engineering import *
from ..preferences import StableDiffusionPreferences
from .control_net import ControlNet

import numpy as np

from functools import reduce

from .. import api
from ..image_utils import bpy_to_np, grayscale

def scheduler_options(self, context):
    return [
        (scheduler, scheduler, '')
        for scheduler in self.get_backend().list_schedulers(context)
    ]

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
    available = ['modify', 'inpaint', 'outpaint']
    return list(filter(lambda x: x[0] in available, init_image_actions))

inpaint_mask_sources = [
    ('alpha', 'Alpha Channel', '', 1),
    ('prompt', 'Prompt', '', 2),
]

def inpaint_mask_sources_filtered(self, context):
    available = ['alpha', 'prompt']
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
        None,
        ('depth_generated', 'Color and Generated Depth', 'Use MiDaS to infer the depth of the initial image and include it in the conditioning. Can give results that more closely match the composition of the source image', 2),
        ('depth_map', 'Color and Depth Map', 'Specify a secondary image to use as the depth map. Can give results that closely match the composition of the depth map', 3),
        ('depth', 'Depth', 'Treat the initial image as a depth map, and ignore any color. Matches the composition of the source image without any color influence', 4),
    ]

def model_options(self, context):
    return [
        None if model is None else (model.id, model.name, model.description)
        for model in self.get_backend().list_models(context)
    ]

def _model_update(self, context):
    options = [m for m in model_options(self, context) if m is not None]
    if self.model == '' and len(options) > 0:
        self.model = options[0]

def backend_options(self, context):
    return [
        (backend._id(), backend.name if hasattr(backend, "name") else backend.__name__, backend.description if hasattr(backend, "description") else "")
        for backend in api.Backend._list_backends()
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
    "backend": EnumProperty(name="Backend", items=backend_options, description="Specify which generation backend to use"),
    "model": EnumProperty(name="Model", items=model_options, description="Specify which model to use for inference", update=_model_update),
    
    "control_nets": CollectionProperty(type=ControlNet),
    "active_control_net": IntProperty(name="Active ControlNet"),

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
    "cfg_scale": FloatProperty(name="CFG Scale", default=7.5, min=0, description="How strongly the prompt influences the image"),
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

def generate_args(self, context, iteration=0, init_image=None, control_images=None) -> api.GenerationArguments:
    is_file_batch = self.prompt_structure == file_batch_structure.id
    file_batch_lines = []
    file_batch_lines_negative = []
    if is_file_batch:
        file_batch_lines = [line.body for line in context.scene.dream_textures_prompt_file.lines if len(line.body.strip()) > 0]
        file_batch_lines_negative = [""] * len(file_batch_lines)
    
    backend: api.Backend = self.get_backend()
    batch_size = backend.get_batch_size(context)
    iteration_limit = len(file_batch_lines) if is_file_batch else self.iterations
    batch_size = min(batch_size, iteration_limit-iteration)

    task: api.Task = api.PromptToImage()
    if self.use_init_img:
        match self.init_img_action:
            case 'modify':
                match self.modify_action_source_type:
                    case 'color':
                        task = api.ImageToImage(
                            image=init_image,
                            strength=self.strength,
                            fit=self.fit
                        )
                    case 'depth_generated':
                        task = api.DepthToImage(
                            depth=None,
                            image=init_image,
                            strength=self.strength
                        )
                    case 'depth_map':
                        task = api.DepthToImage(
                            depth=None if init_image is None else grayscale(bpy_to_np(context.scene.init_depth, color_space=None)),
                            image=init_image,
                            strength=self.strength
                        )
                    case 'depth':
                        task = api.DepthToImage(
                            image=None,
                            depth=None if init_image is None else grayscale(init_image),
                            strength=self.strength
                        )
            case 'inpaint':
                task = api.Inpaint(
                    image=init_image,
                    strength=self.strength,
                    fit=self.fit,
                    mask_source=api.Inpaint.MaskSource.ALPHA if self.inpaint_mask_src == 'alpha' else api.Inpaint.MaskSource.PROMPT,
                    mask_prompt=self.text_mask,
                    confidence=self.text_mask_confidence
                )
            case 'outpaint':
                task = api.Outpaint(
                    image=init_image,
                    origin=(self.outpaint_origin[0], self.outpaint_origin[1])
                )

    return api.GenerationArguments(
        task=task,
        model=next((model for model in self.get_backend().list_models(context) if model is not None and model.id == self.model), None),
        prompt=api.Prompt(
            file_batch_lines[iteration:iteration+batch_size] if is_file_batch else [self.generate_prompt()] * batch_size,
            file_batch_lines_negative[iteration:iteration+batch_size] if is_file_batch else ([self.negative_prompt] * batch_size if self.use_negative_prompt else None)
        ),
        size=(self.width, self.height) if self.use_size else None,
        seed=self.get_seed(),
        steps=self.steps,
        guidance_scale=self.cfg_scale,
        scheduler=self.scheduler,
        seamless_axes=SeamlessAxes(self.seamless_axes),
        step_preview_mode=StepPreviewMode(self.step_preview_mode),
        iterations=self.iterations,
        control_nets=[
            api.models.control_net.ControlNet(
                net.control_net,
                control_images[i] if control_images is not None else None,
                net.conditioning_scale
            )
            for i, net in enumerate(self.control_nets)
            if net.enabled
        ]
    )

def get_backend(self) -> api.Backend:
    return getattr(self, api.Backend._lookup(self.backend)._attribute())

DreamPrompt.generate_prompt = generate_prompt
DreamPrompt.get_prompt_subject = get_prompt_subject
DreamPrompt.get_seed = get_seed
DreamPrompt.generate_args = generate_args
DreamPrompt.get_backend = get_backend