import gc
import logging
import os
from ..models import Checkpoint, ModelConfig, Scheduler

logger = logging.getLogger(__name__)


def revision_paths(model, config="model_index.json"):
    from huggingface_hub.constants import HF_HUB_CACHE

    is_repo = "/" in model
    if os.path.exists(os.path.join(model, config)):
        is_repo = False
    elif not is_repo and os.path.exists(os.path.join(HF_HUB_CACHE, model, config)):
        model = os.path.join(HF_HUB_CACHE, model)
    elif not is_repo:
        raise ValueError(f"{model} is not a valid repo, imported checkpoint, or path")

    if not is_repo:
        return {"main": model}

    model_path = os.path.join(HF_HUB_CACHE, "--".join(["models", *model.split("/")]))
    refs_path = os.path.join(model_path, "refs")
    revisions = {}
    if not os.path.isdir(refs_path):
        return revisions
    for ref in os.listdir(refs_path):
        with open(os.path.join(refs_path, ref)) as f:
            commit_hash = f.read()
        snapshot_path = os.path.join(model_path, "snapshots", commit_hash)
        if os.path.isdir(snapshot_path):
            revisions[ref] = snapshot_path
    return revisions


def cache_check(*, exists_callback=None):
    def decorator(func):
        def wrapper(cache, model, *args, **kwargs):
            if model in cache:
                r = cache[model]
                if exists_callback is not None:
                    r = cache[model] = exists_callback(cache, model, r, *args, **kwargs)
            else:
                r = cache[model] = func(cache, model, *args, **kwargs)
            return r
        return wrapper
    return decorator


@cache_check()
def _load_controlnet_model(cache, model, half_precision):
    from diffusers import ControlNetModel
    import torch

    if isinstance(model, str) and os.path.isfile(model):
        model = Checkpoint(model, None)

    if isinstance(model, Checkpoint):
        control_net_model = ControlNetModel.from_single_file(
            model.path,
            config_file=model.config.original_config if isinstance(model.config, ModelConfig) else model.config,
        )
        if half_precision:
            control_net_model.to(torch.float16)
        return control_net_model

    revisions = revision_paths(model, "config.json")
    if "main" not in revisions:
        # controlnet models shouldn't have a fp16 revision to worry about
        raise FileNotFoundError(f"{model} does not contain a main revision")

    fp16_weights = ["diffusion_pytorch_model.fp16.safetensors", "diffusion_pytorch_model.fp16.bin"]
    fp32_weights = ["diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin"]
    if half_precision:
        weights_names = fp16_weights + fp32_weights
    else:
        weights_names = fp32_weights + fp16_weights

    weights = next((name for name in weights_names if os.path.isfile(os.path.join(revisions["main"], name))), None)
    if weights is None:
        raise FileNotFoundError(f"Can't find appropriate weights in {model}")
    half_weights = weights in fp16_weights
    if not half_precision and half_weights:
        logger.warning(f"Can't load fp32 weights for model {model}, attempting to load fp16 instead")

    return ControlNetModel.from_pretrained(
        revisions["main"],
        torch_dtype=torch.float16 if half_precision else None,
        variant="fp16" if half_weights else None
    )


def _load_checkpoint(model_class, checkpoint, dtype, **kwargs):
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

    if isinstance(checkpoint, Checkpoint):
        model = checkpoint.path
        config = checkpoint.config
    else:
        model = checkpoint
        config = ModelConfig.AUTO_DETECT

    if not os.path.exists(model):
        raise FileNotFoundError(f"Can't locate {model}")

    config_file = config.original_config if isinstance(config, ModelConfig) else config
    if hasattr(model_class, "from_single_file"):
        return model_class.from_single_file(
            model,
            torch_dtype=dtype,
            original_config_file=config_file,
            **kwargs
        )
    else:
        # auto pipelines won't support from_single_file() https://github.com/huggingface/diffusers/issues/4367
        from_pipe = hasattr(model_class, "from_pipe")
        if from_pipe:
            pipeline_class = config.pipeline if isinstance(config, ModelConfig) else None
        else:
            pipeline_class = model_class
        pipe = download_from_original_stable_diffusion_ckpt(
            model,
            from_safetensors=model.endswith(".safetensors"),
            original_config_file=config_file,
            pipeline_class=pipeline_class,
            controlnet=kwargs.get("controlnet", None)
        )
        if dtype is not None:
            pipe.to(torch_dtype=dtype)
        if from_pipe:
            pipe = model_class.from_pipe(pipe, **kwargs)
        return pipe


def _convert_pipe(cache, model, pipe, model_class, half_precision, scheduler, **kwargs):
    if model_class.__name__ not in {
        # some tasks are not supported by auto pipeline
        'DreamTexturesDepth2ImgPipeline',
        'StableDiffusionUpscalePipeline',
    }:
        pipe = model_class.from_pipe(pipe, **kwargs)
    scheduler.create(pipe)
    return pipe


@cache_check(exists_callback=_convert_pipe)
def _load_pipeline(cache, model, model_class, half_precision, scheduler, **kwargs):
    import torch

    dtype = torch.float16 if half_precision else None

    if isinstance(model, Checkpoint) or os.path.splitext(model)[1] in [".ckpt", ".safetensors"]:
        pipe = _load_checkpoint(model_class, model, dtype, **kwargs)
        scheduler.create(pipe)
        return pipe

    revisions = revision_paths(model)
    strategies = []
    if "main" in revisions:
        strategies.append({"model_path": revisions["main"], "variant": "fp16" if half_precision else None})
        if not half_precision:
            # fp16 variant can automatically use fp32 files, but fp32 won't automatically use fp16 files
            strategies.append({"model_path": revisions["main"], "variant": "fp16", "_warn_precision_fallback": True})
    if "fp16" in revisions:
        strategies.append({"model_path": revisions["fp16"], "_warn_precision_fallback": not half_precision})

    if len(strategies) == 0:
        raise FileNotFoundError(f"{model} does not contain a main or fp16 revision")

    exc = None
    for strat in strategies:
        if strat.pop("_warn_precision_fallback", False):
            logger.warning(f"Can't load fp32 weights for model {model}, attempting to load fp16 instead")
        try:
            pipe = model_class.from_pretrained(strat.pop("model_path"), torch_dtype=dtype, safety_checker=None, requires_safety_checker=False, **strat, **kwargs)
            pipe.scheduler = scheduler.create(pipe)
            return pipe
        except Exception as e:
            if exc is None:
                exc = e
    raise exc


def load_model(self, model_class, model, optimizations, scheduler, controlnet=None, sdxl_refiner_model=None, **kwargs):
    import torch
    from diffusers import StableDiffusionXLPipeline, AutoPipelineForImage2Image
    from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

    device = self.choose_device(optimizations)
    half_precision = optimizations.can_use_half(device)
    invalidation_properties = (device, half_precision, optimizations.cpu_offloading(device), controlnet is not None)

    # determine models to be removed from cache
    if not hasattr(self, "_pipe") or self._pipe is None or self._pipe[0] != invalidation_properties:
        model_cache = {}
        self._pipe = (invalidation_properties, model_cache)
        gc.collect()
        torch.cuda.empty_cache()
    else:
        model_cache = self._pipe[1]
        expected_models = {model}
        if sdxl_refiner_model is not None:
            expected_models.add(sdxl_refiner_model)
        if controlnet is not None:
            expected_models.update(name for name in controlnet)
        clear_models = set(model_cache).difference(expected_models)
        for name in clear_models:
            model_cache.pop(name)
        for pipe in model_cache.items():
            if isinstance(getattr(pipe, "controlnet", None), MultiControlNetModel):
                # make sure no longer needed ControlNetModels are cleared
                # the MultiControlNetModel container will be remade
                pipe.controlnet = None
        if len(clear_models) > 0:
            gc.collect()
            torch.cuda.empty_cache()

    # load or obtain models from cache
    if controlnet is not None:
        kwargs["controlnet"] = MultiControlNetModel([
            _load_controlnet_model(model_cache, name, half_precision) for name in controlnet
        ])
    if not isinstance(scheduler, Scheduler):
        try:
            scheduler = Scheduler[scheduler]
        except KeyError:
            raise ValueError(f"scheduler expected one of {[s.name for s in Scheduler]}, got {repr(scheduler)}")
    pipe = _load_pipeline(model_cache, model, model_class, half_precision, scheduler, **kwargs)
    if isinstance(pipe, StableDiffusionXLPipeline) and sdxl_refiner_model is not None:
        return pipe, _load_pipeline(model_cache, sdxl_refiner_model, AutoPipelineForImage2Image, half_precision, scheduler, **kwargs)
    elif sdxl_refiner_model is not None:
        if model_cache.pop(sdxl_refiner_model, None) is not None:
            # refiner was previously used and left enabled but is not compatible with the now selected model
            gc.collect()
            torch.cuda.empty_cache()
        # the caller expects a tuple since refiner was defined
        return pipe, None
    return pipe
