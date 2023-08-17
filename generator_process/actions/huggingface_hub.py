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
from ...api.models.task import *

class ModelType(enum.IntEnum):
    """
    Inferred model type from the U-Net `in_channels`.
    """
    UNKNOWN = 0
    PROMPT_TO_IMAGE = 4
    DEPTH = 5
    UPSCALING = 7
    INPAINTING = 9

    CONTROL_NET = -1

    @classmethod
    def _missing_(cls, _):
        return cls.UNKNOWN
    
    def recommended_model(self) -> str:
        """Provides a recommended model for a given task.

        This method has a bias towards the latest version of official Stability AI models.
        """
        match self:
            case ModelType.PROMPT_TO_IMAGE:
                return "stabilityai/stable-diffusion-2-1"
            case ModelType.DEPTH:
                return "stabilityai/stable-diffusion-2-depth"
            case ModelType.UPSCALING:
                return "stabilityai/stable-diffusion-x4-upscaler"
            case ModelType.INPAINTING:
                return "stabilityai/stable-diffusion-2-inpainting"
            case _:
                return "stabilityai/stable-diffusion-2-1"
    
    def matches_task(self, task: Task) -> bool:
        """Indicates if the model type is correct for a given `Task`.
        
        If not an error should be shown to the user to select a different model.
        """
        match task:
            case PromptToImage():
                return self == ModelType.PROMPT_TO_IMAGE
            case Inpaint():
                return self == ModelType.INPAINTING
            case DepthToImage():
                return self == ModelType.DEPTH
            case Outpaint():
                return self == ModelType.INPAINTING
            case ImageToImage():
                return self == ModelType.PROMPT_TO_IMAGE
            case _:
                return False
    
    @staticmethod
    def from_task(task: Task) -> 'ModelType | None':
        match task:
            case PromptToImage():
                return ModelType.PROMPT_TO_IMAGE
            case Inpaint():
                return ModelType.INPAINTING
            case DepthToImage():
                return ModelType.DEPTH
            case Outpaint():
                return ModelType.INPAINTING
            case ImageToImage():
                return ModelType.PROMPT_TO_IMAGE
            case _:
                return None

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
    from huggingface_hub import HfApi, ModelFilter
    
    if hasattr(self, "huggingface_hub_api"):
        api: HfApi = self.huggingface_hub_api
    else:
        api = HfApi()
        setattr(self, "huggingface_hub_api", api)
    
    filter = ModelFilter(tags="diffusers")
    models = api.list_models(
        filter=filter,
        search=query,
        use_auth_token=token
    )
    return [
        Model(m.modelId, m.author or "", m.tags, m.likes if hasattr(m, "likes") else 0, getattr(m, "downloads", -1), ModelType.UNKNOWN)
        for m in models
        if m.modelId is not None and m.tags is not None and 'diffusers' in (m.tags or {})
    ]

def hf_list_installed_models(self) -> list[Model]:
    from diffusers.utils import DIFFUSERS_CACHE
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

            if os.path.exists(os.path.join(storage_folder, 'model_index.json')):
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
    new_cache_list = list_dir(DIFFUSERS_CACHE)
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

def hf_snapshot_download(
    self,
    model: str,
    token: str,
    variant: str | None = None,
    resume_download=True
):
    from huggingface_hub import utils

    future = Future()
    yield future
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
            future.add_response(DownloadStatus(f"{count} file{'' if count == 1 else 's'}: {self.desc}", self.n, self.total))
    
    from huggingface_hub import file_download
    file_download.tqdm = future_tqdm
    
    from diffusers import StableDiffusionPipeline

    StableDiffusionPipeline.download(
        model,
        use_auth_token=token,
        variant=variant,
        resume_download=resume_download,
    )

    future.set_done()