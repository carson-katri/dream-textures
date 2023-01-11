import functools

import torch

active_dml_patches: list | None = None


def tensor_offload(self, other, *, pre_patch):
    """Fix for operations involving float64 values"""
    try:
        return pre_patch(self, other)
    except RuntimeError:
        device = self.device
        self = self.to("cpu")
        if torch.is_tensor(other):
            other = other.to("cpu")
        return pre_patch(self, other).to(device)


def tensor_ensure_device(self, other, *, pre_patch):
    """Fix for operations where one tensor is DML and the other is CPU"""
    try:
        return pre_patch(self, other)
    except RuntimeError:
        if torch.is_tensor(other) and self.device != other.device:
            if self.device.type != "cpu":
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


def clamp(self, min, max, *, pre_patch):
    if self.dtype == torch.float16 and self.device.type == "privateuseone":
        # clamp() should only be called during float to ubyte image
        # conversion so returning a higher precision doesn't matter.
        # Wrongly converts NaN to max parameter, but that also doesn't matter.
        return torch.clamp(self.to(torch.float32), min, max)
    return pre_patch(self, min, max)


def baddbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None, pre_patch):
    if input.device.type == "privateuseone" and beta == 0:
        if out is not None:
            torch.bmm(batch1, batch2, out=out)
            out *= alpha
            return out
        return alpha * (batch1 @ batch2)
    return pre_patch(input, batch1, batch2, beta=beta, alpha=alpha, out=out)


def enable():
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
    dml_patch_method(torch.Tensor, "__eq__", tensor_offload)
    # LMSDiscreteScheduler.step()
    dml_patch_method(torch.Tensor, "__rsub__", tensor_offload)

    # PNDMScheduler.step()
    dml_patch_method(torch.Tensor, "__mul__", tensor_ensure_device)
    # PNDMScheduler.step()
    dml_patch_method(torch.Tensor, "__sub__", tensor_ensure_device)
    # DDIMScheduler.step() last timestep in image_to_image
    dml_patch_method(torch.Tensor, "__truediv__", tensor_ensure_device)

    # CrossAttention.get_attention_scores()
    # AttentionBlock.forward()
    # Diffusers implementation gives torch.empty() tensors with beta=0 to baddbmm(), which may contain NaNs.
    # DML implementation doesn't properly ignore input argument with beta=0 and causes NaN propagation.
    dml_patch(torch, "baddbmm", baddbmm)

    # float16 patches
    dml_patch(torch, "layer_norm", layer_norm)
    dml_patch(torch, "zeros", zeros)
    dml_patch_method(torch.Tensor, "clamp", clamp)


def disable():
    global active_dml_patches
    if active_dml_patches is None:
        return
    for patch in active_dml_patches:
        setattr(patch["object"], patch["name"], patch["original"])
    active_dml_patches = None
