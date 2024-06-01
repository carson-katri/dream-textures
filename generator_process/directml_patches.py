import functools
import gc

import torch
from torch import Tensor

active_dml_patches: list | None = None


def pad(input, pad, mode="constant", value=None, *, pre_patch):
    if input.device.type == "dml" and mode == "constant":
        pad_dims = torch.tensor(pad, dtype=torch.int32).view(-1, 2).flip(0)
        both_ends = False
        for pre, post in pad_dims:
            if pre != 0 and post != 0:
                both_ends = True
                break
        if both_ends:
            if value is None:
                value = 0
            if pad_dims.size(0) < input.ndim:
                pad_dims = pre_patch(pad_dims, (0, 0, input.ndim-pad_dims.size(0), 0))
            ret = torch.full(torch.Size(torch.tensor(input.size(), dtype=pad_dims.dtype) + pad_dims.sum(dim=1)),
                             fill_value=value, dtype=input.dtype, device=input.device)
            assign_slices = [slice(max(0, int(pre)), None if post <= 0 else -max(0, int(post))) for pre, post in pad_dims]
            index_slices = [slice(max(0, -int(pre)), None if post >= 0 else -max(0, -int(post))) for pre, post in pad_dims]
            ret[assign_slices] = input[index_slices]
            return ret
    return pre_patch(input, pad, mode=mode, value=value)


def layer_norm(input, normalized_shape, weight = None, bias = None, eps = 1e-05, *, pre_patch):
    if input.device.type == "dml":
        return pre_patch(input.contiguous(), normalized_shape, weight, bias, eps)
    return pre_patch(input, normalized_shape, weight, bias, eps)


def retry_OOM(module):
    if hasattr(module, "_retry_OOM"):
        return
    forward = module.forward

    def is_OOM(e: RuntimeError):
        if hasattr(e, "_retry_OOM"):
            return False
        if len(e.args) == 0:
            return False
        if not isinstance(e.args[0], str):
            return False
        return (
            e.args[0].startswith("Could not allocate tensor with") and
            e.args[0].endswith("bytes. There is not enough GPU video memory available!")
        )

    def wrapper(*args, **kwargs):
        try:
            try:
                return forward(*args, **kwargs)
            except RuntimeError as e:
                if is_OOM(e):
                    tb = e.__traceback__.tb_next
                    while tb is not None:
                        # clear locals from traceback so that intermediate tensors can be garbage collected
                        # helps recover from Attention blocks more often
                        tb.tb_frame.clear()
                        tb = tb.tb_next
                    # print("retrying!", type(module).__name__)
                    gc.collect()
                    return forward(*args, **kwargs)
                raise
        except RuntimeError as e:
            if is_OOM(e):
                # only retry leaf modules
                e._retry_OOM = True
            raise

    module.forward = wrapper
    module._retry_OOM = True


def enable(pipe):
    for comp in pipe.components.values():
        if not isinstance(comp, torch.nn.Module):
            continue
        for module in comp.modules():
            retry_OOM(module)

    global active_dml_patches
    if active_dml_patches is not None:
        return
    active_dml_patches = []

    def dml_patch(object, name, patched):
        original = getattr(object, name)
        setattr(object, name, functools.partial(patched, pre_patch=original))
        active_dml_patches.append({"object": object, "name": name, "original": original})

    def dml_patch_method(object, name, patched):
        original = getattr(object, name)
        setattr(object, name, functools.partialmethod(patched, pre_patch=original))
        active_dml_patches.append({"object": object, "name": name, "original": original})

    dml_patch(torch.nn.functional, "pad", pad)

    dml_patch(torch.nn.functional, "layer_norm", layer_norm)

    def decorate_forward(name, module):
        """Helper function to better find which modules DML fails in as it often does
        not raise an exception and immediately crashes the python interpreter."""
        original = module.forward

        def func(self, *args, **kwargs):
            print(f"{name} in module {type(self)}")

            def nan_check(key, x):
                if isinstance(x, Tensor) and x.dtype in [torch.float16, torch.float32] and x.isnan().any():
                    raise RuntimeError(f"{key} got NaN!")

            for i, v in enumerate(args):
                nan_check(i, v)
            for k, v in kwargs.items():
                nan_check(k, v)
            r = original(*args, **kwargs)
            nan_check("return", r)
            return r
        module.forward = func.__get__(module)

    # only enable when testing
    # for name, model in [("text_encoder", pipe.text_encoder), ("unet", pipe.unet), ("vae", pipe.vae)]:
    #     for module in model.modules():
    #         decorate_forward(name, module)


def disable(pipe):
    global active_dml_patches
    if active_dml_patches is None:
        return
    for patch in active_dml_patches:
        setattr(patch["object"], patch["name"], patch["original"])
    active_dml_patches = None
