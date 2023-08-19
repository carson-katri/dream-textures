import logging
import os

logger = logging.getLogger(__name__)


def load_model(self, model_class, model, half_precision, **kwargs):
    import torch
    from diffusers.utils import DIFFUSERS_CACHE

    if not hasattr(self, "_pipe") or self._pipe is None or self._pipe[0] != model:
        dtype = torch.float16 if half_precision else torch.float32
        variant = "fp16" if half_precision else None
        model_path_or_repo = model
        if "/" not in model_path_or_repo:
            model_path_or_repo = os.path.join(DIFFUSERS_CACHE, model)
            if not os.path.isdir(model_path_or_repo):
                raise ValueError(f"{model} is not a valid repo or imported checkpoint")

        try:
            self._pipe = (model, model_class.from_pretrained(model_path_or_repo, torch_dtype=dtype, variant=variant, local_files_only=True, **kwargs))
        except OSError:
            # fp16 variant can automatically use fp32 files, but fp32 won't automatically use fp16 files
            if half_precision:
                raise
            logger.warning(f"Can't load fp32 weights for model {model}, attempting to load fp16 instead")
            self._pipe = (model, model_class.from_pretrained(model_path_or_repo, torch_dtype=dtype, variant="fp16", local_files_only=True, **kwargs))
    elif model_class.__name__ not in {
        # some tasks are not support by auto pipeline
        'DreamTexturesDepth2ImgPipeline'
    }:
        self._pipe = (model, model_class.from_pipe(self._pipe[1], **kwargs))
    return self._pipe[1]