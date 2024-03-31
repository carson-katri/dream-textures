import bpy
from bpy.props import FloatProperty, IntProperty, EnumProperty, BoolProperty
from typing import List

from .api import Backend, StepCallback, Callback
from .api.models import Model, GenerationArguments, GenerationResult
from .api.models.task import PromptToImage, ImageToImage, Inpaint, DepthToImage, Outpaint, Upscale
from .api.models.fix_it_error import FixItError

from .generator_process import Generator
from .generator_process.future import Future
from .generator_process.models import CPUOffload, ModelType, Optimizations, Scheduler

from .preferences import checkpoint_lookup, StableDiffusionPreferences, _template_model_download_progress, InstallModel, model_lookup

from functools import reduce

def _convert_models(models):
    return [
        None if model is None else (model.id, model.name, model.description)
        for model in models
    ]

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

    use_sdxl_refiner: BoolProperty(name="Use SDXL Refiner", default=False, description="Provide a refiner model to run automatically after the initial generation")
    sdxl_refiner_model: EnumProperty(name="SDXL Refiner Model", items=lambda self, context: _convert_models(self.list_models(context)), description="Specify which model to use as a refiner")

    def list_models(self, context):
        def model_case(model, i):
            return Model(
                name=model.model_base.replace('models--', '').replace('--', '/'),
                description=ModelType[model.model_type].name,
                id=model.model_base.replace('models--', '').replace('--', '/')
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
    
    def list_controlnet_models(self, context):
        return [
            Model(
                name=model.model_base.replace('models--', '').replace('--', '/'),
                description="ControlNet",
                id=model.model_base.replace('models--', '').replace('--', '/')
            )
            for model in context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences.installed_models
            if model.model_type == ModelType.CONTROL_NET.name
        ]

    def list_schedulers(self, context) -> List[str]:
        return [scheduler.value for scheduler in Scheduler]

    def get_batch_size(self, context) -> int:
        return self.batch_size

    def optimizations(self) -> Optimizations:
        optimizations = Optimizations()
        for prop in dir(self):
            if hasattr(optimizations, prop) and not prop.startswith('__'):
                setattr(optimizations, prop, getattr(self, prop))
        if self.attention_slice_size_src == 'auto':
            optimizations.attention_slice_size = 'auto'
        optimizations.cpu_offload = CPUOffload(optimizations.cpu_offload)
        return optimizations

    def generate(self, arguments: GenerationArguments, step_callback: StepCallback, callback: Callback):
        gen = Generator.shared()
        common_kwargs = {
            'model': checkpoint_lookup.get(arguments.model.id),
            'scheduler': Scheduler(arguments.scheduler),
            'optimizations': self.optimizations(),
            'prompt': arguments.prompt.positive,
            'steps': arguments.steps,
            'width': arguments.size[0] if arguments.size is not None else None,
            'height': arguments.size[1] if arguments.size is not None else None,
            'seed': arguments.seed,
            'cfg_scale': arguments.guidance_scale,
            'use_negative_prompt': arguments.prompt.negative is not None,
            'negative_prompt': arguments.prompt.negative or "",
            'seamless_axes': arguments.seamless_axes,
            'iterations': arguments.iterations,
            'step_preview_mode': arguments.step_preview_mode,
            
            'sdxl_refiner_model': (checkpoint_lookup.get(self.sdxl_refiner_model) if self.use_sdxl_refiner else None),
        }
        future: Future
        match arguments.task:
            case PromptToImage():
                if len(arguments.control_nets) > 0:
                    future = gen.control_net(
                        **common_kwargs,
                        control_net=[checkpoint_lookup.get(c.model) for c in arguments.control_nets],
                        control=[c.image for c in arguments.control_nets],
                        controlnet_conditioning_scale=[c.strength for c in arguments.control_nets],
                        image=None,
                        inpaint=False,
                        inpaint_mask_src='alpha',
                        text_mask='',
                        text_mask_confidence=1,
                        strength=1
                    )
                else:
                    future = gen.prompt_to_image(**common_kwargs)
            case Inpaint(image=image, fit=fit, strength=strength, mask_source=mask_source, mask_prompt=mask_prompt, confidence=confidence):
                if len(arguments.control_nets) > 0:
                    future = gen.control_net(
                        **common_kwargs,
                        control_net=[c.model for c in arguments.control_nets],
                        control=[c.image for c in arguments.control_nets],
                        controlnet_conditioning_scale=[c.strength for c in arguments.control_nets],
                        image=image,
                        inpaint=True,
                        inpaint_mask_src='alpha' if mask_source == Inpaint.MaskSource.ALPHA else 'prompt',
                        text_mask=mask_prompt,
                        text_mask_confidence=confidence,
                        strength=strength
                    )
                else:
                    future = gen.inpaint(
                        image=image,
                        fit=fit,
                        strength=strength,
                        inpaint_mask_src='alpha' if mask_source == Inpaint.MaskSource.ALPHA else 'prompt',
                        text_mask=mask_prompt,
                        text_mask_confidence=confidence,
                        **common_kwargs
                    )
            case ImageToImage(image=image, strength=strength, fit=fit):
                if len(arguments.control_nets) > 0:
                    future = gen.control_net(
                        **common_kwargs,
                        control_net=[c.model for c in arguments.control_nets],
                        control=[c.image for c in arguments.control_nets],
                        controlnet_conditioning_scale=[c.strength for c in arguments.control_nets],
                        image=image,
                        inpaint=False,
                        inpaint_mask_src='alpha',
                        text_mask='',
                        text_mask_confidence=1,
                        strength=strength
                    )
                else:
                    future = gen.image_to_image(image=image, fit=fit, strength=strength, **common_kwargs)
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
                    outpaint_origin=origin,
                    fit=False,
                    strength=1,
                    inpaint_mask_src='alpha',
                    text_mask='',
                    text_mask_confidence=1,
                    **common_kwargs
                )
            case Upscale(image=image, tile_size=tile_size, blend=blend):
                future = gen.upscale(
                    image=image,
                    tile_size=tile_size,
                    blend=blend,
                    **common_kwargs
                )
            case _:
                raise NotImplementedError()
        def on_step(future: Future, step_image: [GenerationResult]):
            should_continue = step_callback(step_image)
            if not should_continue:
                future.cancel()
                callback(InterruptedError())
        def on_done(future: Future):
            callback(future.result(last_only=True))
        def on_exception(_, exception):
            callback(exception)
        future.add_response_callback(on_step)
        future.add_exception_callback(on_exception)
        future.add_done_callback(on_done)

    def validate(self, arguments: GenerationArguments):
        model = None if arguments.model is None else model_lookup.get(arguments.model.id)
        if model is None:
            raise FixItError("No model selected.", FixItError.ChangeProperty("model"))
        else:
            if not model.model_type.matches_task(arguments.task):
                class DownloadModel(FixItError.Solution):
                    def _draw(self, dream_prompt, context, layout):
                        if not _template_model_download_progress(context, layout):
                            target_model_type = ModelType.from_task(arguments.task)
                            if target_model_type is not None:
                                install_model = layout.operator(InstallModel.bl_idname, text=f"Download {target_model_type.recommended_model()} (Recommended)", icon="IMPORT")
                                install_model.model = target_model_type.recommended_model()
                                install_model.prefer_fp16_revision = context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences.prefer_fp16_revision
                model_task_description = f"""Incorrect model type selected for {type(arguments.task).name().replace('_', ' ').lower()} tasks.
The selected model is for {model.model_type.name.replace('_', ' ').lower()}."""
                if not any(m.model_type.matches_task(arguments.task) for m in model_lookup._models.values()):
                    raise FixItError(
                        message=model_task_description + "\nYou do not have any compatible models downloaded:",
                        solution=DownloadModel()
                    )
                else:
                    raise FixItError(
                        message=model_task_description + "\nSelect a different model below.",
                        solution=FixItError.ChangeProperty("model")
                    )

    def draw_advanced(self, layout, context):
        layout.prop(self, "use_sdxl_refiner")
        col = layout.column()
        col.enabled = self.use_sdxl_refiner
        col.prop(self, "sdxl_refiner_model")

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