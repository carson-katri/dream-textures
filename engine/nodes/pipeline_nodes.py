import bpy
import numpy as np
from dataclasses import dataclass
from typing import Any
import enum
from ..node import DreamTexturesNode
from ...generator_process import Generator
from ...property_groups.control_net import control_net_options
from ...property_groups.dream_prompt import DreamPrompt
from ..annotations import openpose
from ..annotations import depth
from ..annotations import normal
from ..annotations import ade20k
import threading

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
    self.inputs['Source Image'].enabled = self.task in {'image_to_image', 'depth_to_image'}
    self.inputs['Noise Strength'].enabled = self.task in {'image_to_image', 'depth_to_image'}
    if self.task == 'depth_to_image':
        self.inputs['Noise Strength'].default_value = 1.0
    self.inputs['Depth Map'].enabled = self.task == 'depth_to_image'
    self.inputs['ControlNets'].enabled = self.task != 'depth_to_image'
class NodeStableDiffusion(DreamTexturesNode):
    bl_idname = "dream_textures.node_stable_diffusion"
    bl_label = "Stable Diffusion"

    prompt: bpy.props.PointerProperty(type=DreamPrompt)
    task: bpy.props.EnumProperty(name="", items=(
        ('prompt_to_image', 'Prompt to Image', '', 1),
        ('image_to_image', 'Image to Image', '', 2),
        ('depth_to_image', 'Depth to Image', '', 3),
    ), update=_update_stable_diffusion_sockets)

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
        layout.prop(prompt, "pipeline", text="")
        layout.prop(prompt, "model", text="")
        layout.prop(prompt, "scheduler", text="")
        layout.prop(prompt, "seamless_axes", text="")
    
    def execute(self, context, prompt, negative_prompt, width, height, steps, seed, cfg_scale, controlnets, depth_map, source_image, noise_strength):
        self.prompt.use_negative_prompt = True
        self.prompt.negative_prompt = negative_prompt
        self.prompt.steps = steps
        self.prompt.seed = str(seed)
        self.prompt.cfg_scale = cfg_scale
        args = self.prompt.generate_args()

        shared_args = context.depsgraph.scene.dream_textures_engine_prompt.generate_args()

        # the source image is a default color, ignore it.
        if np.array(source_image).shape == (4,):
            source_image = None
        
        if controlnets is not None:
            if not isinstance(controlnets, list):
                controlnets = [controlnets]
            future = Generator.shared().control_net(
                pipeline=args['pipeline'],
                model=args['model'],
                scheduler=args['scheduler'],
                optimizations=shared_args['optimizations'],
                seamless_axes=args['seamless_axes'],
                iterations=args['iterations'],
                step_preview_mode=args['step_preview_mode'],

                control_net=[c.model for c in controlnets],
                control=[c.control(context.depsgraph) for c in controlnets],
                controlnet_conditioning_scale=[c.conditioning_scale for c in controlnets],

                image=np.flipud(np.uint8(source_image * 255)) if self.task in {'image_to_image', 'inpaint'} else None,
                strength=noise_strength,

                inpaint=self.task == 'inpaint',
                inpaint_mask_src='alpha',
                text_mask='',
                text_mask_confidence=1,

                prompt=prompt,
                steps=steps,
                seed=seed,
                width=width,
                height=height,
                cfg_scale=cfg_scale,
                use_negative_prompt=True,
                negative_prompt=negative_prompt
            )
        else:
            match self.task:
                case 'prompt_to_image':
                    future = Generator.shared().prompt_to_image(
                        pipeline=args['pipeline'],
                        model=args['model'],
                        scheduler=args['scheduler'],
                        optimizations=shared_args['optimizations'],
                        seamless_axes=args['seamless_axes'],
                        iterations=args['iterations'],
                        step_preview_mode=args['step_preview_mode'],
                        prompt=prompt,
                        steps=steps,
                        seed=seed,
                        width=width,
                        height=height,
                        cfg_scale=cfg_scale,
                        use_negative_prompt=True,
                        negative_prompt=negative_prompt
                    )
                case 'image_to_image':
                    future = Generator.shared().image_to_image(
                        pipeline=args['pipeline'],
                        model=args['model'],
                        scheduler=args['scheduler'],
                        optimizations=shared_args['optimizations'],
                        seamless_axes=args['seamless_axes'],
                        iterations=args['iterations'],
                        step_preview_mode=args['step_preview_mode'],
                        
                        image=np.uint8(source_image * 255),
                        strength=noise_strength,
                        fit=True,

                        prompt=prompt,
                        steps=steps,
                        seed=seed,
                        width=width,
                        height=height,
                        cfg_scale=cfg_scale,
                        use_negative_prompt=True,
                        negative_prompt=negative_prompt
                    )
                case 'depth_to_image':
                    future = Generator.shared().depth_to_image(
                        pipeline=args['pipeline'],
                        model=args['model'],
                        scheduler=args['scheduler'],
                        optimizations=shared_args['optimizations'],
                        seamless_axes=args['seamless_axes'],
                        iterations=args['iterations'],
                        step_preview_mode=args['step_preview_mode'],
                        
                        depth=depth_map,
                        image=np.uint8(source_image * 255) if source_image is not None else None,
                        strength=noise_strength,

                        prompt=prompt,
                        steps=steps,
                        seed=seed,
                        width=width,
                        height=height,
                        cfg_scale=cfg_scale,
                        use_negative_prompt=True,
                        negative_prompt=negative_prompt
                    )
        event = threading.Event()
        result = None
        exception = None
        def on_response(_, response):
            context.update(response.images[0])
            if context.test_break():
                nonlocal result
                future.cancel()
                result = [response]
                event.set()

        def on_done(future):
            nonlocal result
            result = future.result()
            event.set()
        
        def on_exception(_, error):
            nonlocal exception
            exception = error
            event.set()
        
        future.add_response_callback(on_response)
        future.add_done_callback(on_done)
        future.add_exception_callback(on_exception)
        event.wait()
        if exception is not None:
            raise exception
        return {
            'Image': result[-1].images[-1]
        }

def _update_control_net_sockets(self, context):
    self.inputs['Collection'].enabled = self.input_type == 'collection'
    self.inputs['Image'].enabled = self.input_type == 'image'
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