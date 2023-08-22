import logging
import os

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

    device = self.choose_device(optimizations)
    half_precision = optimizations.can_use_half(device)
    invalidation_properties = (model, device, half_precision, optimizations.cpu_offloading(device))

    if not hasattr(self, "_pipe") or self._pipe is None or self._pipe[0] != invalidation_properties:
        dtype = torch.float16 if half_precision else torch.float32
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