from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Generator, BinaryIO
import copy
import io
import os
import tempfile
import warnings
from contextlib import contextmanager
from functools import partial
from hashlib import sha256
from pathlib import Path
import requests
import json
import enum
from ..future import Future
from ..models import ModelType


@dataclass
class Model:
    id: str
    author: str
    tags: list[str]
    likes: int
    downloads: int
    model_type: ModelType

def hf_list_models(
    self,
    query: str,
    token: str,
) -> list[Model]:
    from huggingface_hub import HfApi
    
    if hasattr(self, "huggingface_hub_api"):
        api: HfApi = self.huggingface_hub_api
    else:
        api = HfApi()
        setattr(self, "huggingface_hub_api", api)

    models = api.list_models(
        tags="diffusers",
        search=query,
        token=token,
    )
    return [
        Model(m.id, m.author or "", m.tags, m.likes if hasattr(m, "likes") else 0, getattr(m, "downloads", -1), ModelType.UNKNOWN)
        for m in models
        if m.id is not None and m.tags is not None and 'diffusers' in (m.tags or {})
    ]

def hf_list_installed_models(self) -> list[Model]:
    from huggingface_hub.constants import HF_HUB_CACHE
    from diffusers.utils.hub_utils import old_diffusers_cache

    def list_dir(cache_dir):
        if not os.path.exists(cache_dir):
            return []

        def detect_model_type(snapshot_folder):
            unet_config = os.path.join(snapshot_folder, 'unet', 'config.json')
            config = os.path.join(snapshot_folder, 'config.json')
            if os.path.exists(unet_config):
                with open(unet_config, 'r') as f:
                    return ModelType(json.load(f)['in_channels'])
            elif os.path.exists(config):
                with open(config, 'r') as f:
                    config_dict = json.load(f)
                    if '_class_name' in config_dict and config_dict['_class_name'] == 'ControlNetModel':
                        return ModelType.CONTROL_NET
                    else:
                        return ModelType.UNKNOWN
            else:
                return ModelType.UNKNOWN

        def _map_model(file):
            storage_folder = os.path.join(cache_dir, file)
            model_type = ModelType.UNKNOWN

            if os.path.exists(os.path.join(storage_folder, 'model_index.json')) or os.path.exists(os.path.join(storage_folder, 'config.json')):
                snapshot_folder = storage_folder
                model_type = detect_model_type(snapshot_folder)
            else:
                refs_folder = os.path.join(storage_folder, "refs")
                if not os.path.exists(refs_folder):
                    return None
                for revision in os.listdir(refs_folder):
                    ref_path = os.path.join(storage_folder, "refs", revision)
                    with open(ref_path) as f:
                        commit_hash = f.read()
                    snapshot_folder = os.path.join(storage_folder, "snapshots", commit_hash)
                    if (detected_type := detect_model_type(snapshot_folder)) != ModelType.UNKNOWN:
                        model_type = detected_type
                        break

            return Model(
                storage_folder,
                "",
                [],
                -1,
                -1,
                model_type
            )
        return [
            model for model in (
                _map_model(file) for file in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, file))
            )
            if model is not None
        ]
    new_cache_list = list_dir(HF_HUB_CACHE)
    model_ids = [os.path.basename(m.id) for m in new_cache_list]
    for model in list_dir(old_diffusers_cache):
        if os.path.basename(model.id) not in model_ids:
            new_cache_list.append(model)
    return new_cache_list

@dataclass
class DownloadStatus:
    file: str
    index: int
    total: int

    @classmethod
    def hook_download_tqdm(cls, future):
        from huggingface_hub import utils, file_download
        progresses = set()

        class future_tqdm(utils.tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.progress()

            def update(self, n=1):
                ret = super().update(n=n)
                self.progress()
                return ret

            def progress(self):
                nonlocal progresses
                progresses.add(self)
                ratio = self.n / self.total
                count = 0
                for tqdm in progresses:
                    r = tqdm.n / tqdm.total
                    if r == 1:
                        continue
                    count += 1
                    if tqdm != self and ratio < r:
                        # only show download status of most complete file
                        return
                future.add_response(cls(f"{count} file{'' if count == 1 else 's'}: {self.desc}", self.n, self.total))
        file_download.tqdm = future_tqdm

def hf_snapshot_download(
    self,
    model: str,
    token: str,
    variant: str | None = None,
    resume_download=True
):
    from huggingface_hub import snapshot_download, repo_info
    from diffusers import StableDiffusionPipeline
    from diffusers.pipelines.pipeline_utils import variant_compatible_siblings

    future = Future()
    yield future
    DownloadStatus.hook_download_tqdm(future)

    info = repo_info(model, token=token)
    files = [file.rfilename for file in info.siblings]

    if "model_index.json" in files:
        # check if the variant files are available before trying to download them
        _, variant_files = variant_compatible_siblings(files, variant=variant)
        StableDiffusionPipeline.download(
            model,
            token=token,
            variant=variant if len(variant_files) > 0 else None,
            resume_download=resume_download,
        )
    elif "config.json" in files:
        # individual model, such as controlnet or vae

        fp16_weights = ["diffusion_pytorch_model.fp16.safetensors", "diffusion_pytorch_model.fp16.bin"]
        fp32_weights = ["diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin"]
        if variant == "fp16":
            weights_names = fp16_weights + fp32_weights
        else:
            weights_names = fp32_weights + fp16_weights

        weights = next((name for name in weights_names if name in files), None)
        if weights is None:
            raise FileNotFoundError(f"Can't find appropriate weights in {model}")

        snapshot_download(
            model,
            token=token,
            resume_download=resume_download,
            allow_patterns=["config.json", weights]
        )
    else:
        raise ValueError(f"{model} doesn't appear to be a pipeline or model")

    future.set_done()
