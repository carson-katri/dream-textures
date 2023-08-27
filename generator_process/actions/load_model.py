import logging
import os
from ..models import Checkpoint, ModelConfig

logger = logging.getLogger(__name__)


def revision_paths(model, config="model_index.json"):
    from diffusers.utils import DIFFUSERS_CACHE

    is_repo = "/" in model
    if os.path.exists(os.path.join(model, config)):
        is_repo = False
    elif not is_repo and os.path.exists(os.path.join(DIFFUSERS_CACHE, model, config)):
        model = os.path.join(DIFFUSERS_CACHE, model)
    elif not is_repo:
        raise ValueError(f"{model} is not a valid repo, imported checkpoint, or path")

    if not is_repo:
        return {"main": model}

    model_path = os.path.join(DIFFUSERS_CACHE, "--".join(["models", *model.split("/")]))
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


def load_checkpoint(model_class, checkpoint, dtype, **kwargs):
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
            original_config_file=config_file
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


def load_model(self, model_class, model, optimizations, **kwargs):
    import torch

    device = self.choose_device(optimizations)
    half_precision = optimizations.can_use_half(device)
    invalidation_properties = (model, device, half_precision, optimizations.cpu_offloading(device))
    reload_pipeline = not hasattr(self, "_pipe") or self._pipe is None or self._pipe[0] != invalidation_properties
    dtype = torch.float16 if half_precision else None

    if reload_pipeline and (isinstance(model, Checkpoint) or os.path.splitext(model)[1] in [".ckpt", ".safetensors"]):
        self._pipe = (invalidation_properties, load_checkpoint(model_class, model, dtype, **kwargs))
    elif reload_pipeline:
        revisions = revision_paths(model)

        strategies = []
        if "main" in revisions:
            strategies.append({"model_path": revisions["main"], "variant": "fp16" if half_precision else None})
            if not half_precision:
                # fp16 variant can automatically use fp32 files, but fp32 won't automatically use fp16 files
                strategies.append({"model_path": revisions["main"], "variant": "fp16", "_warn_precision_fallback": True})
        if "fp16" in revisions:
            strategies.append({"model_path": revisions["fp16"], "_warn_precision_fallback": not half_precision})

        def try_strategies():
            exc = None
            for strat in strategies:
                if strat.pop("_warn_precision_fallback", False):
                    logger.warning(f"Can't load fp32 weights for model {model}, attempting to load fp16 instead")
                try:
                    return model_class.from_pretrained(strat.pop("model_path"), torch_dtype=dtype, **strat, **kwargs)
                except Exception as e:
                    if exc is None:
                        exc = e
            raise exc
        self._pipe = (invalidation_properties, try_strategies())
    elif model_class.__name__ not in {
        # some tasks are not support by auto pipeline
        'DreamTexturesDepth2ImgPipeline'
    }:
        self._pipe = (invalidation_properties, model_class.from_pipe(self._pipe[1], **kwargs))
    return self._pipe[1]