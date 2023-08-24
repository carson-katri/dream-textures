import logging
import os
from .huggingface_hub import checkpoint_links
from .convert_original_stable_diffusion_to_diffusers import ModelConfig

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


def load_model(self, model_class, model, optimizations, **kwargs):
    import torch
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

    device = self.choose_device(optimizations)
    half_precision = optimizations.can_use_half(device)
    invalidation_properties = (model, device, half_precision, optimizations.cpu_offloading(device))
    reload_pipeline = not hasattr(self, "_pipe") or self._pipe is None or self._pipe[0] != invalidation_properties
    basename = os.path.basename(model)
    filename, extension = os.path.splitext(basename)
    dtype = torch.float16 if half_precision else torch.float32

    if reload_pipeline and extension in [".ckpt", ".safetensors"]:
        path = model
        config = ModelConfig.AUTO_DETECT
        if not os.path.exists(model):
            path = None
            for p, c in checkpoint_links.items():
                if os.path.isfile(p) and os.path.basename(p) == basename:
                    path = p
                    config = c
                    break
                elif os.path.isdir(p) and basename in os.listdir(p):
                    path = os.path.join(p, basename)
                    config = c
                    # only break for a direct checkpoint link to allow overriding config in a directory link
            if path is None:
                raise FileNotFoundError(f"Can't locate {model}")

        if hasattr(model_class, "from_single_file"):
            self._pipe = (invalidation_properties, model_class.from_single_file(
                path,
                torch_dtype=dtype,
                original_config_file=config.original_config
            ))
        elif hasattr(model_class, "from_pipe"):
            # auto pipelines won't support from_single_file() https://github.com/huggingface/diffusers/issues/4367
            pipe = download_from_original_stable_diffusion_ckpt(
                path,
                from_safetensors=extension == ".safetensors",
                original_config_file=config.original_config,
                pipeline_class=config.pipeline
            )
            pipe.to(torch_dtype=dtype)
            self._pipe = (invalidation_properties, model_class.from_pipe(pipe))
        else:
            raise NotImplementedError(f"Can't load {model} for {model_class}, does not have from_single_file() or from_pipe()")
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