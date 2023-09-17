import math
from typing import Optional

import numpy as np
from ..actions.detect_seamless import SeamlessAxes
from numpy.typing import NDArray


class UpscaleTiler:
    def __init__(
            self,
            image: NDArray,
            scale: int,
            tile_size: int | tuple[int, int],
            blend: int | tuple[int, int],
            seamless_axes: SeamlessAxes,
            defer_seamless: bool = True,
            out_channels: Optional[int] = None
    ):
        height, width = image.shape[:2]
        if scale < 1:
            raise ValueError("scale must be 1 or higher")
        if isinstance(tile_size, int):
            tile_size = (tile_size, tile_size)
        if tile_size[0] <= 0 or tile_size[1] <= 0:
            raise ValueError("tile size must be 1 or higher")
        if isinstance(blend, int):
            blend = (blend, blend)
        if blend[0] < 0 or blend[1] < 0:
            raise ValueError("blend must be 0 or higher")
        seamless_axes = SeamlessAxes(seamless_axes)
        if defer_seamless:
            # Seamless handling may be deferred to upscaler model or VAE rather than using larger or multiple tiles
            seamless_axes = SeamlessAxes((seamless_axes.x and width > tile_size[0], seamless_axes.y and height > tile_size[1]))
        max_width = width*2 if seamless_axes.x else width
        max_height = height*2 if seamless_axes.y else height
        tile_size = (min(tile_size[0], max_width), min(tile_size[1], max_height))
        blend = (min(blend[0], math.ceil(tile_size[0]/2)), min(blend[1], math.ceil(tile_size[1]/2)))
        self.image = image
        self.scale = scale
        self.tile_size = tile_size
        self.blend = blend
        self.seamless_axes = seamless_axes
        self.x_tiles = self.axis_tiles(width, tile_size[0], blend[0], seamless_axes.x)
        self.y_tiles = self.axis_tiles(height, tile_size[1], blend[1], seamless_axes.y)
        if out_channels is None:
            out_channels = image.shape[2]
        # combined image with last channel containing pixel weights
        self.canvas = np.zeros((image.shape[0] * scale, image.shape[1] * scale, out_channels + 1), dtype=np.float32)

        scaled_tile_size = (tile_size[0] * scale, tile_size[1] * scale)
        weight_gradient_y = [min(i + 1, scaled_tile_size[1] - i) for i in range(scaled_tile_size[1])]
        weight_gradient_x = [min(i + 1, scaled_tile_size[0] - i) for i in range(scaled_tile_size[0])]
        tile_weight = np.zeros(scaled_tile_size, dtype=np.float32)
        tile_weight[:] = weight_gradient_y
        # determines how much each pixel in a blended area influences the final color, basically a pyramid
        self.tile_weight = np.minimum(tile_weight, np.reshape(weight_gradient_x, (scaled_tile_size[0], 1)))

    @staticmethod
    def axis_tiles(axis_size: int, tile_size: int, blend: int, seamless: bool) -> list[int]:
        """
        Returns a list of values where each tile starts on an axis.
        Blend is only guaranteed as a minimum and may vary by a pixel between tiles.
        """
        if seamless:
            count = math.ceil(axis_size / (tile_size - blend))
            blend_balance = math.ceil(tile_size - axis_size / count)
            final = min(axis_size - tile_size + blend_balance, axis_size * 2 - tile_size)
        else:
            count = math.ceil((axis_size - tile_size) / (tile_size - blend)) + 1
            final = axis_size - tile_size
        if count == 1:
            return [0]
        return [i * final // (count - 1) for i in range(count)]

    def combined(self) -> NDArray:
        return self.canvas[:, :, :-1]

    def index_to_xy(self, index: int):
        key_y = index % len(self.y_tiles)
        key_x = (index - key_y) // len(self.y_tiles)
        return key_x, key_y

    def __getitem__(self, key: int | tuple[int, int]) -> NDArray:
        if isinstance(key, int):
            key = self.index_to_xy(key)
        image = self.image
        tile_size = self.tile_size
        x0 = self.x_tiles[key[0]]
        x1 = x0 + tile_size[0]
        x2 = image.shape[1] - x0
        y0 = self.y_tiles[key[1]]
        y1 = y0 + tile_size[1]
        y2 = image.shape[0] - y0
        if x2 >= tile_size[0] and y2 >= tile_size[1]:
            return image[y0:y1, x0:x1]
        # seamless axis wrapping
        if isinstance(image, np.ndarray):
            tile = np.empty((tile_size[0], tile_size[1], image.shape[2]), dtype=image.dtype)
        else:
            import torch
            tile = torch.empty((tile_size[0], tile_size[1], image.shape[2]), dtype=image.dtype, device=image.device)
        if x2 < tile_size[0]:
            if y2 < tile_size[1]:
                # wrap bottom/right to top/left
                tile[:y2, :x2] = image[y0:, x0:]
                tile[y2:, :x2] = image[:tile_size[1] - y2, x0:]
                tile[:y2, x2:] = image[y0:, :tile_size[0] - x2]
                tile[y2:, x2:] = image[:tile_size[1] - y2, :tile_size[0] - x2]
            else:
                # wrap right to left
                tile[:, :x2] = image[y0:y1, x0:]
                tile[:, x2:] = image[y0:y1, :tile_size[0] - x2]
        else:
            # wrap bottom to top
            tile[:y2] = image[y0:, x0:x1]
            tile[y2:] = image[:tile_size[1] - y2, x0:x1]
        return tile

    def __setitem__(self, key: int | tuple[int, int], tile: NDArray):
        if isinstance(key, int):
            key = self.index_to_xy(key)
        canvas = self.canvas
        scale = self.scale
        tile_size = (self.tile_size[0] * scale, self.tile_size[1] * scale)
        tile_weight = self.tile_weight
        x0 = self.x_tiles[key[0]] * scale
        x1 = x0 + tile_size[0]
        x2 = canvas.shape[1] - x0
        y0 = self.y_tiles[key[1]] * scale
        y1 = y0 + tile_size[1]
        y2 = canvas.shape[0] - y0

        def update(canvas_slice, tile_slice, weight_slice):
            weight_slice = weight_slice.reshape(weight_slice.shape[0], weight_slice.shape[1], 1)
            # undo weighted average, then add new tile with its weights applied and average again
            canvas_slice[:, :, :-1] *= canvas_slice[:, :, -1:]
            canvas_slice[:, :, :-1] += tile_slice * weight_slice
            canvas_slice[:, :, -1:] += weight_slice
            canvas_slice[:, :, :-1] /= canvas_slice[:, :, -1:]

        if x2 >= tile_size[0] and y2 >= tile_size[1]:
            update(canvas[y0:y1, x0:x1], tile, tile_weight)
        elif x2 < tile_size[0]:
            if y2 < tile_size[1]:
                update(canvas[y0:, x0:], tile[:y2, :x2], tile_weight[:y2, :x2])
                update(canvas[:tile_size[1] - y2, x0:], tile[y2:, :x2], tile_weight[y2:, :x2])
                update(canvas[y0:, :tile_size[0] - x2], tile[:y2, x2:], tile_weight[:y2, x2:])
                update(canvas[:tile_size[1] - y2, :tile_size[0] - x2], tile[y2:, x2:], tile_weight[y2:, x2:])
            else:
                update(canvas[y0:y1, x0:], tile[:, :x2], tile_weight[:, :x2])
                update(canvas[y0:y1, :tile_size[0] - x2], tile[:, x2:], tile_weight[:, x2:])
        else:
            update(canvas[y0:, x0:x1], tile[:y2], tile_weight[:y2])
            update(canvas[:tile_size[1] - y2, x0:x1], tile[y2:], tile_weight[y2:])

    def __iter__(self):
        for x in range(len(self.x_tiles)):
            for y in range(len(self.y_tiles)):
                yield (x, y), self[x, y]

    def __len__(self):
        return len(self.x_tiles) * len(self.y_tiles)


def tiled_decode_latents(self, latents, return_dict=False, *, pre_patch, optimizations):
    # not all pipelines (namely upscale) have the vae_scale_factor attribute
    vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    default_size = self.unet.config.sample_size * vae_scale_factor
    match optimizations.vae_tiling:
        case "full":
            tile_size = default_size
            blend = math.ceil(tile_size / 8)
        case "half":
            tile_size = math.ceil(default_size / 2)
            blend = math.ceil(tile_size / 8)
        case "manual":
            tile_size = optimizations.vae_tile_size
            blend = optimizations.vae_tile_blend
        case _:
            return pre_patch(latents)

    seamless_axes = getattr(self.vae, "seamless_axes", SeamlessAxes.OFF)

    images = []
    for image_latents in latents.split(1, dim=0):
        tiler = UpscaleTiler(
            image_latents.squeeze(0).permute(1, 2, 0),
            vae_scale_factor,
            math.ceil(tile_size / vae_scale_factor),
            math.ceil(blend / vae_scale_factor),
            seamless_axes,
            out_channels=self.vae.config.out_channels
        )

        configure_model_padding(self.vae, seamless_axes & ~tiler.seamless_axes)

        for id, tile in tiler:
            tiler[id] = pre_patch(tile.permute(2, 0, 1).unsqueeze(0)).sample.squeeze(0).permute(1, 2, 0).cpu().numpy()
        images.append(np.expand_dims(tiler.combined(), 0).transpose(0, 3, 1, 2))
    configure_model_padding(self.vae, seamless_axes)
    images = np.concatenate(images)
    import torch
    images = torch.from_numpy(images)
    if not return_dict:
        return (images,)
    from diffusers.models.vae import DecoderOutput
    return DecoderOutput(images)

def configure_model_padding(model, seamless_axes):
    import torch.nn as nn
    """
    Modifies the 2D convolution layers to use a circular padding mode based on the `seamless_axes` option.
    """
    seamless_axes = SeamlessAxes(seamless_axes)
    if seamless_axes == SeamlessAxes.AUTO:
        seamless_axes = seamless_axes.OFF
    if getattr(model, "seamless_axes", SeamlessAxes.OFF) == seamless_axes:
        return
    model.seamless_axes = seamless_axes
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

def _conv_forward_asymmetric(self, input, weight, bias):
    import torch.nn as nn
    """
    Patch for Conv2d._conv_forward that supports asymmetric padding
    """
    if input.device.type == "dml":
        # DML pad() will wrongly fill the tensor in constant mode with the supplied value
        # (default 0) when padding on both ends of a dimension, can't split to two calls.
        working = nn.functional.pad(input, self._reversed_padding_repeated_twice, mode='circular')
        pad_w0, pad_w1, pad_h0, pad_h1 = self._reversed_padding_repeated_twice
        if self.asymmetric_padding_mode[0] == 'constant':
            working[:, :, :, :pad_w0] = 0
            if pad_w1 > 0:
                working[:, :, :, -pad_w1:] = 0
        if self.asymmetric_padding_mode[1] == 'constant':
            working[:, :, :pad_h0] = 0
            if pad_h1 > 0:
                working[:, :, -pad_h1:] = 0
    else:
        working = nn.functional.pad(input, self.asymmetric_padding[0], mode=self.asymmetric_padding_mode[0])
        working = nn.functional.pad(working, self.asymmetric_padding[1], mode=self.asymmetric_padding_mode[1])
    return nn.functional.conv2d(working, weight, bias, self.stride, nn.modules.utils._pair(0), self.dilation, self.groups)
