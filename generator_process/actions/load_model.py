import logging
import os

logger = logging.getLogger(__name__)


def load_model(self, model_class, model, half_precision, **kwargs):
    import torch
    from diffusers.utils import DIFFUSERS_CACHE

    if not hasattr(self, "_pipe") or self._pipe is None or self._pipe[0] != model:
        dtype = torch.float16 if half_precision else torch.float32
        model_path_or_repo = model
        is_repo = "/" in model_path_or_repo
        if os.path.exists(os.path.join(model, "model_index.json")):
            is_repo = False
        elif not is_repo and os.path.exists(os.path.join(DIFFUSERS_CACHE, model, "model_index.json")):
            model_path_or_repo = os.path.join(DIFFUSERS_CACHE, model)
        elif not is_repo:
            raise ValueError(f"{model} is not a valid repo, imported checkpoint, or path")

        strategies = [{"variant": "fp16" if half_precision else None}]
        if not half_precision:
            # fp16 variant can automatically use fp32 files, but fp32 won't automatically use fp16 files
            strategies.append({"variant": "fp16", "_warn_precision_fallback": True})
        if is_repo and os.path.exists(os.path.join(DIFFUSERS_CACHE, "--".join(["models", *model.split("/")]), "refs", "fp16")):
            strategies.append({"revision": "fp16"})

        def try_strategies():
            exc = None
            for strat in strategies:
                if strat.pop("_warn_precision_fallback", False):
                    logger.warning(f"Can't load fp32 weights for model {model}, attempting to load fp16 instead")
                try:
                    return model_class.from_pretrained(model_path_or_repo, torch_dtype=dtype, local_files_only=True, **strat, **kwargs)
                except Exception as e:
                    if exc is None:
                        exc = e
            raise exc
        self._pipe = (model, try_strategies())
    elif model_class.__name__ not in {
        # some tasks are not support by auto pipeline
        'DreamTexturesDepth2ImgPipeline'
    }:
        self._pipe = (model, model_class.from_pipe(self._pipe[1], **kwargs))
    return self._pipe[1]