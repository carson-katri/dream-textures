from typing import Callable

from .actor import Actor, is_actor_process

class RunInSubprocess(Exception):
    """
    Decorators to support running functions that are not defined under the Generator class in its subprocess.
    This is to reduce what would otherwise be duplicate function definitions that logically don't belong to
    the Generator, but require something in its subprocess (such as access to installed dependencies).
    """

    def __new__(cls, func=None):
        if func is None:
            # support `raise RunInSubprocess`
            return super().__new__(cls)
        return cls.always(func)

    @staticmethod
    def always(func):
        if is_actor_process:
            return func
        def wrapper(*args, **kwargs):
            return Generator.shared().call(wrapper, *args, **kwargs).result()
        RunInSubprocess._copy_attributes(func, wrapper)
        return wrapper

    @staticmethod
    def when(condition: bool | Callable[..., bool]):
        if not isinstance(condition, Callable):
            if condition:
                return RunInSubprocess.always
            return lambda x: x
        def decorator(func):
            if is_actor_process:
                return func
            def wrapper(*args, **kwargs):
                if condition(*args, **kwargs):
                    return Generator.shared().call(wrapper, *args, **kwargs).result()
                return func(*args, **kwargs)
            RunInSubprocess._copy_attributes(func, wrapper)
            return wrapper
        return decorator

    @staticmethod
    def when_raised(func):
        if is_actor_process:
            return func
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RunInSubprocess:
                return Generator.shared().call(wrapper, *args, **kwargs).result()
        RunInSubprocess._copy_attributes(func, wrapper)
        return wrapper

    @staticmethod
    def _copy_attributes(src, dst):
        for n in ["__annotations__", "__doc__", "__name__", "__module__", "__qualname__"]:
            if hasattr(src, n):
                setattr(dst, n, getattr(src, n))

class Generator(Actor):
    """
    The actor used for all background processes.
    """

    from .actions.choose_device import choose_device
    from .actions.load_model import load_model
    from .actions.prompt_to_image import prompt_to_image
    from .actions.image_to_image import image_to_image
    from .actions.inpaint import inpaint
    from .actions.outpaint import outpaint
    from .actions.upscale import upscale
    from .actions.depth_to_image import depth_to_image
    from .actions.control_net import control_net
    from .actions.huggingface_hub import hf_snapshot_download, hf_list_models, hf_list_installed_models
    from .actions.convert_original_stable_diffusion_to_diffusers import convert_original_stable_diffusion_to_diffusers
    from .actions.detect_seamless import detect_seamless
    from .actions.controlnet_aux import controlnet_aux

    @staticmethod
    def call(func, *args, **kwargs):
        return func(*args, **kwargs)
