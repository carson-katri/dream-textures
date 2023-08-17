def load_model(self, model_class, model, half_precision, **kwargs):
    import torch
    if not hasattr(self, "_pipe") or self._pipe is None or self._pipe[0] != model:
        self._pipe = (model, model_class.from_pretrained(model, torch_dtype=torch.float16 if half_precision else torch.float32, variant="fp16", **kwargs))
    elif model_class.__name__ not in {
        # some tasks are not support by auto pipeline
        'DreamTexturesDepth2ImgPipeline'
    }:
        self._pipe = (model, model_class.from_pipe(self._pipe[1], **kwargs))
    return self._pipe[1]