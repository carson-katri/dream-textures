from ..models import *
from ..actions.detect_seamless import SeamlessAxes

class CachedPipeline:
    """A pipeline that has been cached for subsequent runs."""

    pipeline: Any
    """The diffusers pipeline to re-use"""

    invalidation_properties: tuple
    """Values that, when changed, will invalid this cached pipeline"""

    snapshot_folder: str
    """The snapshot folder containing the model"""

    def __init__(self, pipeline: Any, invalidation_properties: tuple, snapshot_folder: str):
        self.pipeline = pipeline
        self.invalidation_properties = invalidation_properties
        self.snapshot_folder = snapshot_folder

    def is_valid(self, properties: tuple):
        return properties == self.invalidation_properties

def load_pipe(self, action, generator_pipeline, model, optimizations, scheduler, device):
    """
    Use a cached pipeline, or create the pipeline class and cache it.
    
    The cached pipeline will be invalidated if the model or use_cpu_offload options change.
    """
    import torch
    import gc

    invalidation_properties = (
        action, model, device,
        optimizations.can_use("sequential_cpu_offload", device),
        optimizations.can_use("half_precision", device),
    )
    cached_pipe: CachedPipeline = self._cached_pipe if hasattr(self, "_cached_pipe") else None
    if cached_pipe is not None and cached_pipe.is_valid(invalidation_properties):
        pipe = cached_pipe.pipeline
    else:
        # Release the cached pipe before loading the new one.
        if cached_pipe is not None:
            del self._cached_pipe
            del cached_pipe
            gc.collect()

        revision = "fp16" if optimizations.can_use_half(device) else None
        snapshot_folder = model_snapshot_folder(model, revision)
        pipe = generator_pipeline.from_pretrained(
            snapshot_folder,
            revision=revision,
            torch_dtype=torch.float16 if optimizations.can_use_half(device) else torch.float32,
        )
        pipe = pipe.to(device)
        setattr(self, "_cached_pipe", CachedPipeline(pipe, invalidation_properties, snapshot_folder))
        cached_pipe = self._cached_pipe
    if 'scheduler' in os.listdir(cached_pipe.snapshot_folder):
        pipe.scheduler = scheduler.create(pipe, {
            'model_path': cached_pipe.snapshot_folder,
            'subfolder': 'scheduler',
        })
    else:
        pipe.scheduler = scheduler.create(pipe, None)
    return pipe

def model_snapshot_folder(model, preferred_revision: str | None = None):
    """ Try to find the preferred revision, but fallback to another revision if necessary. """
    import diffusers
    storage_folder = os.path.join(diffusers.utils.DIFFUSERS_CACHE, model)
    if os.path.exists(os.path.join(storage_folder, 'model_index.json')): # converted model
        snapshot_folder = storage_folder
    else: # hub model
        revisions = {}
        for revision in os.listdir(os.path.join(storage_folder, "refs")):
            ref_path = os.path.join(storage_folder, "refs", revision)
            with open(ref_path) as f:
                commit_hash = f.read()

            snapshot_folder = os.path.join(storage_folder, "snapshots", commit_hash)
            if len(os.listdir(snapshot_folder)) > 1:
                revisions[revision] = snapshot_folder

        if len(revisions) == 0:
            return None
        elif preferred_revision in revisions:
            revision = preferred_revision
        elif preferred_revision in [None, "fp16"] and "main" in revisions:
            revision = "main"
        elif preferred_revision in [None, "main"] and "fp16" in revisions:
            revision = "fp16"
        else:
            revision = next(iter(revisions.keys()))
        snapshot_folder = revisions[revision]

    return snapshot_folder

def configure_model_padding(model, seamless_axes):
    import torch.nn as nn
    """
    Modifies the 2D convolution layers to use a circular padding mode based on the `seamless` and `seamless_axes` options.
    """
    seamless_axes = SeamlessAxes(seamless_axes)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if seamless_axes.x or seamless_axes.y:
                m.asymmetric_padding_mode = (
                    'circular' if seamless_axes.x else 'constant',
                    'circular' if seamless_axes.y else 'constant'
                )
                m.asymmetric_padding = (
                    (m._reversed_padding_repeated_twice[0], m._reversed_padding_repeated_twice[1], 0, 0),
                    (0, 0, m._reversed_padding_repeated_twice[2], m._reversed_padding_repeated_twice[3])
                )
                m._conv_forward = _conv_forward_asymmetric.__get__(m, nn.Conv2d)
            else:
                m._conv_forward = nn.Conv2d._conv_forward.__get__(m, nn.Conv2d)
                if hasattr(m, 'asymmetric_padding_mode'):
                    del m.asymmetric_padding_mode
                if hasattr(m, 'asymmetric_padding'):
                    del m.asymmetric_padding