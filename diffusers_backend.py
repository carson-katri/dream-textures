from bpy.props import FloatProperty, IntProperty, EnumProperty, BoolProperty
from typing import List, Tuple

from .api import Backend, StepCallback, Callback
from .api.models import Model, Task, Prompt, SeamlessAxes, GenerationResult, StepPreviewMode
from .api.models.task import PromptToImage, ImageToImage, Inpaint, DepthToImage, Outpaint

from .generator_process import Generator
from .generator_process.actions.prompt_to_image import ImageGenerationResult
from .generator_process.future import Future
from .generator_process.models import Optimizations, Scheduler
from .generator_process.actions.huggingface_hub import ModelType
from .preferences import StableDiffusionPreferences

from functools import reduce

class DiffusersBackend(Backend):
    name = "HuggingFace Diffusers"
    description = "Local image generation inside of Blender"

    attention_slicing: BoolProperty(name="Attention Slicing", default=True, description="Computes attention in several steps. Saves some memory in exchange for a small speed decrease")
    attention_slice_size_src: EnumProperty(
        name="Attention Slice Size",
        items=(
            ("auto", "Automatic", "Computes attention in two steps", 1),
            ("manual", "Manual", "Computes attention in `attention_head_dim // size` steps. A smaller `size` saves more memory.\n"
                                "`attention_head_dim` must be a multiple of `size`, otherwise the image won't generate properly.\n"
                                "`attention_head_dim` can be found within the model snapshot's unet/config.json file", 2)
        ),
        default=1
    )
    attention_slice_size: IntProperty(name="Attention Slice Size", default=1, min=1)
    cudnn_benchmark: BoolProperty(name="cuDNN Benchmark", description="Allows cuDNN to benchmark multiple convolution algorithms and select the fastest", default=False)
    tf32: BoolProperty(name="TF32", description="Utilizes tensor cores on Ampere (RTX 30xx) or newer GPUs for matrix multiplications.\nHas no effect if half precision is enabled", default=False)
    half_precision: BoolProperty(name="Half Precision", description="Reduces memory usage and increases speed in exchange for a slight loss in image quality.\nHas no effect if CPU only is enabled or using a GTX 16xx GPU", default=True)
    cpu_offload: EnumProperty(
        name="CPU Offload",
        items=(
            ("off", "Off", "", 0),
            ("model", "Model", "Some memory savings with minimal speed penalty", 1),
            ("submodule", "Submodule", "Better memory savings with large speed penalty", 2)
        ),
        default=0,
        description="Dynamically moves models in and out of device memory for reduced memory usage with reduced speed"
    )
    channels_last_memory_format: BoolProperty(name="Channels Last Memory Format", description="An alternative way of ordering NCHW tensors that may be faster or slower depending on the device", default=False)
    sdp_attention: BoolProperty(
        name="SDP Attention",
        description="Scaled dot product attention requires less memory and often comes with a good speed increase.\n"
                    "Prompt recall may not produce the exact same image, but usually only minor noise differences.\n"
                    "Overrides attention slicing",
        default=True
    )
    batch_size: IntProperty(name="Batch Size", default=1, min=1, description="Improves speed when using iterations or upscaling in exchange for higher memory usage.\nHighly recommended to use with VAE slicing enabled")
    vae_slicing: BoolProperty(name="VAE Slicing", description="Reduces memory usage of batched VAE decoding. Has no effect if batch size is 1.\nMay have a small performance improvement with large batches", default=True)
    vae_tiling: EnumProperty(
        name="VAE Tiling",
        items=(
            ("off", "Off", "", 0),
            ("half", "Half", "Uses tiles of half the selected model's default size. Likely to cause noticeably inaccurate colors", 1),
            ("full", "Full", "Uses tiles of the selected model's default size, intended for use where image size is manually set higher. May cause slightly inaccurate colors", 2),
            ("manual", "Manual", "", 3)
        ),
        default=0,
        description="Decodes generated images in tiled regions to reduce memory usage in exchange for longer decode time and less accurate colors.\nCan allow for generating larger images that would otherwise run out of memory on the final step"
    )
    vae_tile_size: IntProperty(name="VAE Tile Size", min=1, default=512, description="Width and height measurement of tiles. Smaller sizes are more likely to cause inaccurate colors and other undesired artifacts")
    vae_tile_blend: IntProperty(name="VAE Tile Blend", min=0, default=64, description="Minimum amount of how much each edge of a tile will intersect its adjacent tile")
    cfg_end: FloatProperty(name="CFG End", min=0, max=1, default=1, description="The percentage of steps to complete before disabling classifier-free guidance")
    cpu_only: BoolProperty(name="CPU Only", default=False, description="Disables GPU acceleration and is extremely slow")

    def list_models(self, context):
        def model_case(model, i):
            return Model(
                name=model.model_base.replace('models--', '').replace('--', '/'),
                description=ModelType[model.model_type].name,
                id=model.model_base
            )
        models = {}
        for i, model in enumerate(context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences.installed_models):
            if model.model_type in {ModelType.CONTROL_NET.name, ModelType.UNKNOWN.name}:
                continue
            if model.model_type not in models:
                models[model.model_type] = [model_case(model, i)]
            else:
                models[model.model_type].append(model_case(model, i))
        return reduce(
            lambda a, b: a + [None] + sorted(b, key=lambda m: m.id),
            [
                models[group]
                for group in sorted(models.keys())
            ],
            []
        )

    def list_schedulers(self, context) -> List[str]:
        return [scheduler.value for scheduler in Scheduler]

    def optimizations(self) -> Optimizations:
        optimizations = Optimizations()
        for prop in dir(self):
            if hasattr(optimizations, prop):
                setattr(optimizations, prop, getattr(self, prop))
        if self.attention_slice_size_src == 'auto':
            optimizations.attention_slice_size = 'auto'
        return optimizations

    def generate(self, task: Task, model: Model, prompt: Prompt, size: Tuple[int, int] | None, seed: int, steps: int, guidance_scale: float, scheduler: str, seamless_axes: SeamlessAxes, step_preview_mode: StepPreviewMode, iterations: int, step_callback: StepCallback, callback: Callback):
        gen = Generator.shared()
        common_kwargs = {
            'model': model.id,
            'scheduler': Scheduler(scheduler),
            'optimizations': self.optimizations(),
            'prompt': prompt.positive,
            'steps': steps,
            'width': size[0] if size is not None else None,
            'height': size[1] if size is not None else None,
            'seed': seed,
            'cfg_scale': guidance_scale,
            'use_negative_prompt': prompt.negative is not None,
            'negative_prompt': prompt.negative or "",
            'seamless_axes': seamless_axes,
            'iterations': iterations,
            'step_preview_mode': step_preview_mode,
        }
        future: Future
        match task:
            case PromptToImage():
                print(common_kwargs)
                import pickle
                del common_kwargs['optimizations'].__annotations__
                print(pickle.dumps(common_kwargs))
                future = gen.prompt_to_image(**common_kwargs)
            case ImageToImage(image=image, strength=strength, fit=fit):
                future = gen.image_to_image(image=image, fit=fit, strength=strength, **common_kwargs)
            case Inpaint(image=image, fit=fit, strength=strength, mask_source=mask_source, mask_prompt=mask_prompt, confidence=confidence):
                future = gen.inpaint(
                    image=image,
                    fit=fit,
                    strength=strength,
                    inpaint_mask_src='alpha' if mask_source == Inpaint.MaskSource.ALPHA else 'prompt',
                    text_mask=mask_prompt,
                    text_mask_confidence=confidence,
                    **common_kwargs
                )
            case DepthToImage(depth=depth, image=image, strength=strength):
                future = gen.depth_to_image(
                    depth=depth,
                    image=image,
                    strength=strength,
                    **common_kwargs
                )
            case Outpaint(image=image, origin=origin):
                future = gen.outpaint(
                    image=image,
                    width=size[0] if size is not None else None,
                    height=size[1] if size is not None else None,
                    outpaint_origin=origin,
                    **common_kwargs
                )
            case _:
                raise NotImplementedError()
        def on_step(_, step_image: ImageGenerationResult):
            step_callback(GenerationResult(image=step_image.images[-1], seed=step_image.seeds[-1]))
        def on_done(future: Future):
            result: ImageGenerationResult = future.result(last_only=True)
            callback([
                GenerationResult(image=result.images[i], seed=result.seeds[i])
                for i in range(len(result.images))
            ])
        def on_exception(_, exception):
            callback(exception)
        future.add_response_callback(on_step)
        future.add_exception_callback(on_exception)
        future.add_done_callback(on_done)

    def draw_speed_optimizations(self, layout, context):
        inferred_device = Optimizations.infer_device()
        if self.cpu_only:
            inferred_device = "cpu"
        def optimization(prop):
            if Optimizations.device_supports(prop, inferred_device):
                layout.prop(self, prop)

        optimization("cudnn_benchmark")
        optimization("tf32")
        optimization("half_precision")
        optimization("channels_last_memory_format")
        optimization("batch_size")
        optimization("cfg_end")
    
    def draw_memory_optimizations(self, layout, context):
        inferred_device = Optimizations.infer_device()
        if self.cpu_only:
            inferred_device = "cpu"
        def optimization(prop):
            if Optimizations.device_supports(prop, inferred_device):
                layout.prop(self, prop)

        optimization("sdp_attention")
        optimization("attention_slicing")
        slice_size_row = layout.row()
        slice_size_row.prop(self, "attention_slice_size_src")
        if self.attention_slice_size_src == 'manual':
            slice_size_row.prop(self, "attention_slice_size", text="Size")
        optimization("cpu_offload")
        optimization("cpu_only")
        optimization("vae_slicing")
        optimization("vae_tiling")
        if self.vae_tiling == "manual":
            optimization("vae_tile_size")
            optimization("vae_tile_blend")