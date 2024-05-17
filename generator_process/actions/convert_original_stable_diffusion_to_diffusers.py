import os

from .huggingface_hub import DownloadStatus
from ..future import Future
from ..models import ModelConfig


def convert_original_stable_diffusion_to_diffusers(
    self,
    checkpoint_path: str,
    model_config: ModelConfig,
    half_precision: bool,
) -> str:
    import torch
    from huggingface_hub.constants import HF_HUB_CACHE
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt, download_controlnet_from_original_ckpt

    future = Future()
    yield future
    DownloadStatus.hook_download_tqdm(future)

    future.add_response(DownloadStatus(f"Reading {checkpoint_path}", 0, 1))
    index = 0
    def hook_save_pretrained(model, dirs_count, total):
        old_save_pretrained = model.save_pretrained
        def save_pretrained(self, save_directory, *args, **kwargs):
            nonlocal index
            dirs = []
            directory = save_directory
            for _ in range(dirs_count):
                dirs.append(os.path.basename(directory))
                directory = os.path.dirname(directory)
            dirs.reverse()
            future.add_response(DownloadStatus(f"Saving {os.path.join(*dirs)}", index, total))
            index += 1
            return old_save_pretrained(save_directory, *args, **kwargs)
        model.save_pretrained = save_pretrained.__get__(model)

    if model_config in [ModelConfig.CONTROL_NET_1_5, ModelConfig.CONTROL_NET_2_1]:
        pipe = download_controlnet_from_original_ckpt(
            checkpoint_path,
            original_config_file=model_config.original_config,
            from_safetensors=checkpoint_path.endswith(".safetensors"),
        )
        if half_precision:
            pipe.to(dtype=torch.float16)
        index = 1
        hook_save_pretrained(pipe, 1, 2)
    else:
        pipe = download_from_original_stable_diffusion_ckpt(
            checkpoint_path,
            original_config_file=model_config.original_config,
            from_safetensors=checkpoint_path.endswith(".safetensors"),
            pipeline_class=model_config.pipeline
        )
        if half_precision:
            pipe.to(torch_dtype=torch.float16)
        models = []
        for name in pipe._get_signature_keys(pipe)[0]:
            model = getattr(pipe, name, None)
            if model is not None and hasattr(model, "save_pretrained"):
                models.append(model)
        for i, model in enumerate(models):
            hook_save_pretrained(model, 2, len(models))
    dump_path = os.path.join(HF_HUB_CACHE, os.path.splitext(os.path.basename(checkpoint_path))[0])
    pipe.save_pretrained(dump_path, variant="fp16" if half_precision else None)
    future.set_done()
