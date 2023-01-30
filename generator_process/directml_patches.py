import functools

import torch
from torch import Tensor

active_dml_patches: list | None = None


def tensor_ensure_device(self, other, *, pre_patch):
    """Fix for operations where one tensor is DML and the other is CPU."""
    if isinstance(other, Tensor) and self.device != other.device:
        if self.device.type != "cpu":
            other = other.to(self.device)
        else:
            self = self.to(other.device)
    return pre_patch(self, other)


def baddbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None, pre_patch):
    if input.device.type == "privateuseone" and beta == 0:
        if out is not None:
            torch.bmm(batch1, batch2, out=out)
            out *= alpha
            return out
        return alpha * (batch1 @ batch2)
    return pre_patch(input, batch1, batch2, beta=beta, alpha=alpha, out=out)


def pad(input, pad, mode="constant", value=None, *, pre_patch):
    if input.device.type == "privateuseone" and mode == "constant":
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


def getitem(self, key, *, pre_patch):
    if isinstance(key, Tensor) and "privateuseone" in [self.device.type, key.device.type] and key.numel() == 1:
        return pre_patch(self, int(key))
    return pre_patch(self, key)


def enable(pipe):
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

    # Not all places where the patches have an effect are necessarily listed.

    # PNDMScheduler.step()
    dml_patch_method(Tensor, "__mul__", tensor_ensure_device)
    # PNDMScheduler.step()
    dml_patch_method(Tensor, "__sub__", tensor_ensure_device)
    # DDIMScheduler.step() last timestep in image_to_image
    dml_patch_method(Tensor, "__truediv__", tensor_ensure_device)

    # CrossAttention.get_attention_scores()
    # AttentionBlock.forward()
    # Diffusers implementation gives torch.empty() tensors with beta=0 to baddbmm(), which may contain NaNs.
    # DML implementation doesn't properly ignore input argument with beta=0 and causes NaN propagation.
    dml_patch(torch, "baddbmm", baddbmm)

    dml_patch(torch.nn.functional, "pad", pad)
    # DDIMScheduler.step(), PNDMScheduler.step(), No error messages or crashes, just may randomly freeze.
    dml_patch_method(Tensor, "__getitem__", getitem)

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
            return original(*args, **kwargs)
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
