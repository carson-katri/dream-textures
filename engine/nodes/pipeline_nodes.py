import bpy
import numpy as np
from dataclasses import dataclass
from typing import Any, List
import enum
from ..node import DreamTexturesNode
from ...generator_process import Generator
from ...property_groups.control_net import control_net_options
from ...property_groups.dream_prompt import DreamPrompt
from ..annotations import openpose
from ..annotations import depth
from ..annotations import normal
from ..annotations import ade20k
from ... import api
from ...property_groups.seamless_result import SeamlessAxes
import threading
from ... import image_utils

class NodeSocketControlNet(bpy.types.NodeSocket):
    bl_idname = "NodeSocketControlNet"
    bl_label = "ControlNet Socket"

    def __init__(self):
        self.link_limit = 0

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return (0.63, 0.63, 0.63, 1)

class ControlType(enum.IntEnum):
    DEPTH = 1
    OPENPOSE = 2
    NORMAL = 3
    ADE20K_SEGMENTATION = 4

@dataclass
class ControlNet:
    model: str
    image: Any
    collection: Any
    control_type: ControlType
    conditioning_scale: float

    def control(self, context):
        if self.image is not None:
            return np.flipud(self.image)
        else:
            match self.control_type:
                case ControlType.DEPTH:
                    return np.flipud(depth.render_depth_map(context, collection=self.collection))
                case ControlType.OPENPOSE:
                    return np.flipud(openpose.render_openpose_map(context, collection=self.collection))
                case ControlType.NORMAL:
                    return np.flipud(normal.render_normal_map(context, collection=self.collection))
                case ControlType.ADE20K_SEGMENTATION:
                    return np.flipud(ade20k.render_ade20k_map(context, collection=self.collection))

def _update_stable_diffusion_sockets(self, context):
    inputs = {socket.name: socket for socket in self.inputs}
    inputs['Source Image'].enabled = self.task in {'image_to_image', 'depth_to_image', 'inpaint'}
    inputs['Noise Strength'].enabled = self.task in {'image_to_image', 'depth_to_image'}
    if self.task == 'depth_to_image':
        inputs['Noise Strength'].default_value = 1.0
    inputs['Depth Map'].enabled = self.task == 'depth_to_image'
    inputs['ControlNets'].enabled = self.task != 'depth_to_image'
class NodeStableDiffusion(DreamTexturesNode):
    bl_idname = "dream_textures.node_stable_diffusion"
    bl_label = "Stable Diffusion"

    prompt: bpy.props.PointerProperty(type=DreamPrompt)
    task: bpy.props.EnumProperty(name="", items=(
        ('prompt_to_image', 'Prompt to Image', '', 1),
        ('image_to_image', 'Image to Image', '', 2),
        ('depth_to_image', 'Depth to Image', '', 3),
        ('inpaint', 'Inpaint', '', 4),
    ), update=_update_stable_diffusion_sockets)

    def update(self):
        self.prompt.backend = bpy.context.scene.dream_textures_render_engine.backend

    def init(self, context):
        self.inputs.new("NodeSocketColor", "Depth Map")
        self.inputs.new("NodeSocketColor", "Source Image")
        self.inputs.new("NodeSocketFloat", "Noise Strength").default_value = 0.75

        self.inputs.new("NodeSocketString", "Prompt")
        self.inputs.new("NodeSocketString", "Negative Prompt")

        self.inputs.new("NodeSocketInt", "Width").default_value = 512
        self.inputs.new("NodeSocketInt", "Height").default_value = 512
        
        self.inputs.new("NodeSocketInt", "Steps").default_value = 25
        self.inputs.new("NodeSocketInt", "Seed")
        self.inputs.new("NodeSocketFloat", "CFG Scale").default_value = 7.50
        
        self.inputs.new("NodeSocketControlNet", "ControlNets")

        self.outputs.new("NodeSocketColor", "Image")

        _update_stable_diffusion_sockets(self, context)

    def draw_buttons(self, context, layout):
        layout.prop(self, "task")
        prompt = self.prompt
        layout.prop(prompt, "model", text="")
        layout.prop(prompt, "scheduler", text="")
        layout.prop(prompt, "seamless_axes", text="")
    
    def execute(self, context, prompt, negative_prompt, width, height, steps, seed, cfg_scale, controlnets, depth_map, source_image, noise_strength):
        backend: api.Backend = self.prompt.get_backend()

        if np.array(source_image).shape == (4,):
            # the source image is a default color, ignore it.
            source_image = None
        else:
            source_image = image_utils.color_transform(np.flipud(source_image), "Linear", "sRGB")

        def get_task():
            match self.task:
                case 'prompt_to_image':
                    return api.PromptToImage()
                case 'image_to_image':
                    return api.ImageToImage(source_image, noise_strength, fit=False)
                case 'depth_to_image':
                    return api.DepthToImage(image_utils.grayscale(depth_map), source_image, noise_strength)
                case 'inpaint':
                    return api.Inpaint(source_image, noise_strength, fit=False, mask_source=api.Inpaint.MaskSource.ALPHA, mask_prompt="", confidence=0)
        
        def map_controlnet(c):
            return api.models.control_net.ControlNet(c.model, c.control(context.depsgraph), c.conditioning_scale)

        args = api.GenerationArguments(
            get_task(),
            model=next(model for model in self.prompt.get_backend().list_models(context) if model is not None and model.id == self.prompt.model),
            prompt=api.Prompt(
                prompt,
                negative_prompt
            ),
            size=(width, height),
            seed=seed,
            steps=steps,
            guidance_scale=cfg_scale,
            scheduler=self.prompt.scheduler,
            seamless_axes=SeamlessAxes(self.prompt.seamless_axes),
            step_preview_mode=api.models.StepPreviewMode.FAST,
            iterations=1,
            control_nets=[map_controlnet(c) for c in controlnets] if isinstance(controlnets, list) else ([map_controlnet(controlnets)] if controlnets is not None else [])
        )
        
        event = threading.Event()
        result = None
        exception = None
        def step_callback(progress: List[api.GenerationResult]) -> bool:
            context.update(image_utils.image_to_np(progress[-1].image, default_color_space="sRGB", to_color_space="Linear", top_to_bottom=False))
            return True
            # if context.test_break():
            #     nonlocal result
            #     result = [response]
            #     event.set()

        def callback(results: List[api.GenerationResult] | Exception):
            if isinstance(results, Exception):
                nonlocal exception
                exception = results
                event.set()
            else:
                nonlocal result
                result = image_utils.image_to_np(results[-1].image, default_color_space="sRGB", to_color_space="Linear", top_to_bottom=False)
                event.set()
        
        backend = self.prompt.get_backend()
        backend.generate(args, step_callback=step_callback, callback=callback)

        event.wait()
        if exception is not None:
            raise exception
        return {
            'Image': result
        }

def _update_control_net_sockets(self, context):
    inputs = {socket.name: socket for socket in self.inputs}
    inputs['Collection'].enabled = self.input_type == 'collection'
    inputs['Image'].enabled = self.input_type == 'image'
class NodeControlNet(DreamTexturesNode):
    bl_idname = "dream_textures.node_control_net"
    bl_label = "ControlNet"

    control_net: bpy.props.EnumProperty(name="", items=control_net_options)
    input_type: bpy.props.EnumProperty(name="", items=(
        ('collection', 'Collection', '', 1),
        ('image', 'Image', '', 2),
    ), update=_update_control_net_sockets)
    control_type: bpy.props.EnumProperty(name="", items=(
        ('DEPTH', 'Depth', '', 1),
        ('OPENPOSE', 'OpenPose', '', 2),
        ('NORMAL', 'Normal Map', '', 3),
        ('ADE20K_SEGMENTATION', 'ADE20K Segmentation', '', 4),
    ))

    def init(self, context):
        self.inputs.new("NodeSocketCollection", "Collection")
        self.inputs.new("NodeSocketColor", "Image")
        self.inputs.new("NodeSocketFloat", "Conditioning Scale").default_value = 1

        self.outputs.new(NodeSocketControlNet.bl_idname, "Control")

        _update_control_net_sockets(self, context)

    def draw_buttons(self, context, layout):
        layout.prop(self, "control_net")
        layout.prop(self, "input_type")
        if self.input_type != 'image':
            layout.prop(self, "control_type")
    
    def execute(self, context, collection, image, conditioning_scale):
        return {
            'Control': ControlNet(
                self.control_net,
                image if self.input_type == 'image' else None,
                collection if self.input_type == 'collection' else None,
                ControlType[self.control_type],
                conditioning_scale
            )
        }