from dataclasses import dataclass
import os

@dataclass
class Model:
    id: str
    author: str
    tags: list[str]
    likes: int
    downloads: int

def hf_list_models(
    self,
    query: str
) -> list[Model]:
    from huggingface_hub import HfApi, ModelFilter
    
    if hasattr(self, "huggingface_hub_api"):
        api: HfApi = self.huggingface_hub_api
    else:
        api = HfApi()
        setattr(self, "huggingface_hub_api", api)
    
    filter = ModelFilter(tags="diffusers", task="text-to-image")
    models = api.list_models(
        filter=filter,
        search=query
    )

    return list(map(lambda m: Model(m.modelId, m.author, m.tags, m.likes, getattr(m, "downloads", -1)), models))

def hf_list_installed_models(self) -> list[Model]:
    from diffusers.utils import DIFFUSERS_CACHE
    if not os.path.exists(DIFFUSERS_CACHE):
        return []
    return list(
        filter(
            lambda x: os.path.isdir(x.id),
            map(lambda x: Model(os.path.join(DIFFUSERS_CACHE, x), "", [], -1, -1), os.listdir(DIFFUSERS_CACHE))
        )
    )

def hf_snapshot_download(
    self,
    model: str,
    token: str
) -> None:
    from huggingface_hub import snapshot_download
    from diffusers import StableDiffusionPipeline
    from diffusers.utils import DIFFUSERS_CACHE, WEIGHTS_NAME, CONFIG_NAME, ONNX_WEIGHTS_NAME
    from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
    from diffusers.hub_utils import http_user_agent
    config_dict = StableDiffusionPipeline.get_config_dict(
        model,
        cache_dir=DIFFUSERS_CACHE,
        resume_download=True,
        force_download=False,
        use_auth_token=token
    )
    # make sure we only download sub-folders and `diffusers` filenames
    folder_names = [k for k in config_dict.keys() if not k.startswith("_")]
    allow_patterns = [os.path.join(k, "*") for k in folder_names]
    allow_patterns += [WEIGHTS_NAME, SCHEDULER_CONFIG_NAME, CONFIG_NAME, ONNX_WEIGHTS_NAME, StableDiffusionPipeline.config_name]

    # make sure we don't download flax weights
    ignore_patterns = "*.msgpack"

    requested_pipeline_class = config_dict.get("_class_name", StableDiffusionPipeline.__name__)
    user_agent = {"pipeline_class": requested_pipeline_class}
    user_agent = http_user_agent(user_agent)

    # download all allow_patterns
    cached_folder = snapshot_download(
        model,
        cache_dir=DIFFUSERS_CACHE,
        resume_download=True,
        use_auth_token=token,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        user_agent=user_agent,
    )
    return 