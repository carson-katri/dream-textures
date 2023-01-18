import functools

import torch
from torch import Tensor

active_dml_patches: list | None = None


def tensor_offload(self, other, *, pre_patch):
    """Fix for operations involving float64 values"""
    try:
        return pre_patch(self, other)
    except RuntimeError:
        device = self.device
        self = self.to("cpu")
        if isinstance(other, Tensor):
            other = other.to("cpu")
        return pre_patch(self, other).to(device)


def tensor_ensure_device(self, other, *, pre_patch):
    """Fix for operations where one tensor is DML and the other is CPU.
    Or when DML gets confused by one tensor having an index and the other doesn't."""
    try:
        return pre_patch(self, other)
    except RuntimeError:
        if isinstance(other, Tensor) and self.device != other.device:
            if self.device.type == "privateuseone" and other.device.type == "privateuseone":
                if self.device.index is None:
                    self = self.to(other.device)
                else:
                    other = other.to(self.device)
            elif self.device.type != "cpu":
                other = other.to(self.device)
            else:
                self = self.to(other.device)
            return pre_patch(self, other)
        raise


def layer_norm(input, normalized_shape, weight, bias, eps, cudnn_enabled, *, pre_patch):
    if input.device.type == "privateuseone" and input.dtype == torch.float16:
        return pre_patch(input.to(torch.float32), normalized_shape, weight.to(torch.float32),
                         bias.to(torch.float32), eps, cudnn_enabled).to(torch.float16)
    return pre_patch(input, normalized_shape, weight, bias, eps, cudnn_enabled)


def group_norm(input, num_groups, weight, bias, eps, cudnn_enabled, *, pre_patch):
    return pre_patch(input.to('cpu').to(torch.float32), num_groups, weight.to('cpu').to(torch.float32),
                     bias.to('cpu').to(torch.float32), eps, cudnn_enabled).to(input.dtype).to(input.device)

    # Alternative implementations that cause some artifacts.
    # Works okay for unet but not vae, regardless of which device it runs on.
    # There should only be minimal float error differences between the two...

    # N, C, H, W = input.size()
    # x = input.view(N, num_groups, C // num_groups, H, W)
    # # The operator 'aten::var_mean.correction' is not currently supported on the DML backend and will fall back to run on the CPU.
    # var, mean = torch.var_mean(x, [2, 3, 4], unbiased=False, keepdim=True)
    # x = (x - mean) / torch.sqrt(var + eps)
    # return x.view(N, C, H, W) * weight.view(1, C, 1, 1) + bias.view(1, C, 1, 1)

    # # 100% DML float32 capable and mostly in-place operations for less memory allocation
    # N, C, H, W = input.size()
    # x = input.view(N, num_groups, C // num_groups, H, W)
    # mean = torch.mean(x, [2, 3, 4], keepdim=True)
    # var = torch.var(x, [2, 3, 4], unbiased=True, keepdim=True)
    # x = x.detach().clone().sub_(mean).div_(var.add_(eps).sqrt_())
    # x = x.view(N, C, H, W).mul_(weight.view(1, C, 1, 1)).add_(bias.view(1, C, 1, 1))
    # # sub = pre_patch(input.to('cpu'), num_groups, weight.to('cpu'), bias.to('cpu'), eps, cudnn_enabled)-x.to('cpu')
    # # print(torch.min(sub), torch.max(sub))
    # return x


def device_is(device, device_type):
    if device is None:
        return 'cpu' == device_type
    device = torch.device(device)
    return device.type == device_type


def zeros(*size, out=None, dtype=None, layout=torch.strided, device=None, pre_patch):
    if dtype == torch.float16 and device_is(device, "privateuseone"):
        if out is None:
            out = torch.empty(*size, dtype=dtype, layout=layout, device=device)
        # [F D:\a\_work\1\s\pytorch-directml-plugin\torch_directml\csrc\engine\dml_util.cc:73] Check failed: false
        # out[:] = 0
        out[:] = torch.tensor(0, dtype=dtype, device=device)
        return out
    return pre_patch(*size, out=out, dtype=dtype, layout=layout, device=device)


def clamp(self, min=None, max=None, *, out=None, pre_patch):
    if self.dtype == torch.float16 and self.device.type == "privateuseone":
        return pre_patch(self.to(torch.float32), min, max, out=out).to(torch.float16)
    return pre_patch(self, min, max, out=out)


def tensor_clamp(self, min=None, max=None, *, pre_patch):
    if self.dtype == torch.float16 and self.device.type == "privateuseone":
        return pre_patch(self.to(torch.float32), min, max).to(torch.float16)
    return pre_patch(self, min, max)


def baddbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None, pre_patch):
    if input.device.type == "privateuseone" and beta == 0:
        if out is not None:
            torch.bmm(batch1, batch2, out=out)
            out *= alpha
            return out
        return alpha * (batch1 @ batch2)
    return pre_patch(input, batch1, batch2, beta=beta, alpha=alpha, out=out)


def pow(self, other, *, pre_patch):
    if self.device.type == "privateuseone" and self.dtype == torch.float16 and isinstance(other, (int, float)):
        if other == 0.5:
            return self.sqrt()
        return self.to(torch.float32).pow_(other).to(torch.float16)
    return pre_patch(self, other)


def pad(input, pad, mode="constant", value=None, *, pre_patch):
    if input.device.type == "privateuseone" and mode == "constant":
        pad_dims = torch.tensor(pad, dtype=torch.int32).view(-1, 2).flip(0)
        both_ends = False
        for pre, post in pad_dims:
            if pre != 0 and post != 0:
                both_ends = True
                break
        if input.dtype == torch.float16 or both_ends:
            if value is None:
                value = 0
            value = torch.tensor(value, dtype=input.dtype, device=input.device)
            if pad_dims.size(0) < input.ndim:
                pad_dims = pre_patch(pad_dims, (0, 0, input.ndim-pad_dims.size(0), 0))
            ret = torch.empty(torch.Size(torch.tensor(input.size(), dtype=pad_dims.dtype) + pad_dims.sum(dim=1)),
                              dtype=input.dtype, device=input.device)
            ret[:] = value
            assign_slices = [slice(max(0, int(pre)), None if post <= 0 else -max(0, int(post))) for pre, post in pad_dims]
            index_slices = [slice(max(0, -int(pre)), None if post >= 0 else -max(0, -int(post))) for pre, post in pad_dims]
            ret[assign_slices] = input[index_slices]
            return ret
    return pre_patch(input, pad, mode=mode, value=value)


def getitem(self, key, *, pre_patch):
    if isinstance(key, Tensor) and "privateuseone" in [self.device.type, key.device.type] and key.numel() == 1:
        return pre_patch(self, int(key))
    return pre_patch(self, key)


def compare(self, other, *, pre_patch):
    if self.device.type == "privateuseone" and self.dtype == torch.float16:
        if isinstance(other, Tensor) and other.dtype == torch.float16:
            other = other.to(torch.float32)
        return pre_patch(self.to(torch.float32), other)
    return pre_patch(self, other)


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

    dml_patch(torch, "group_norm", group_norm)

    # LMSDiscreteScheduler.scale_model_input()
    dml_patch_method(Tensor, "__eq__", tensor_offload)
    # LMSDiscreteScheduler.step()
    dml_patch_method(Tensor, "__rsub__", tensor_offload)

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

    # float16 patches
    dml_patch(torch, "layer_norm", layer_norm)
    dml_patch(torch, "zeros", zeros)
    dml_patch(torch, "clamp", clamp)
    dml_patch_method(Tensor, "clamp", tensor_clamp)
    dml_patch_method(Tensor, "__pow__", pow)
    dml_patch(torch.nn.functional, "pad", pad)
    # DDIMScheduler.step(), PNDMScheduler.step(), No error messages or crashes, just may randomly freeze.
    dml_patch_method(Tensor, "__getitem__", getitem)
    # EulerDiscreteScheduler.step()
    dml_patch_method(Tensor, "__gt__", compare)
    dml_patch_method(Tensor, "__ge__", compare)
    dml_patch_method(Tensor, "__lt__", compare)
    dml_patch_method(Tensor, "__le__", compare)

    def decorate_forward(name, module):
        """Helper function to better find which modules DML fails in as it often does
        not raise an exception and immediately crashes the python interpreter."""
        original = module.forward

        def func(self, *args, **kwargs):
            print(f"{name} in module {type(self)}")

            def nan_check(key, x):
                if isinstance(x, Tensor) and x.cpu().isnan().any():
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
