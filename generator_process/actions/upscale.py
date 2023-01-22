import math
import numpy as np
from .prompt_to_image import Optimizations, Scheduler, StepPreviewMode
from .detect_seamless import SeamlessAxes
import random
from dataclasses import dataclass
from numpy.typing import NDArray

@dataclass
class ImageUpscaleResult:
    image: NDArray | None
    tile: int
    total: int
    final: bool


class UpscaleTiler:
    def __init__(self, image: NDArray, scale: int, tile_size: int, blend: int, seamless_axes: SeamlessAxes):
        self.image = image
        self.scale = scale
        self.tile_size = tile_size
        self.blend = blend
        seamless_axes = SeamlessAxes(seamless_axes)
        self.x_tiles = self.axis_tiles(image.shape[1], tile_size, blend, seamless_axes.x)
        self.y_tiles = self.axis_tiles(image.shape[0], tile_size, blend, seamless_axes.y)
        # combined image with last channel containing pixel weights
        self.canvas = np.zeros((image.shape[0] * scale, image.shape[1] * scale, image.shape[2] + 1), dtype=np.float32)

        scaled_tile_size = tile_size * scale
        weight_gradient = [min(i + 1, scaled_tile_size - i) for i in range(scaled_tile_size)]
        tile_weight = np.zeros((scaled_tile_size, scaled_tile_size), dtype=np.float32)
        tile_weight[:] = weight_gradient
        # determines how much each pixel in a blended area influences the final color, basically a pyramid
        self.tile_weight = np.minimum(tile_weight, np.reshape(weight_gradient, (scaled_tile_size, 1)))

    @staticmethod
    def axis_tiles(axis_size: int, tile_size: int, blend: int, seamless: bool) -> list[int]:
        """
        Returns a list of values where each tile starts on an axis.
        Blend is only guaranteed as a minimum and may vary by a pixel between tiles.
        """
        if seamless:
            count = math.ceil(axis_size / (tile_size - blend))
            blend_balance = math.ceil(tile_size - axis_size / count)
            final = axis_size - tile_size + blend_balance
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
        x1 = x0 + tile_size
        x2 = image.shape[1] - x0
        y0 = self.y_tiles[key[1]]
        y1 = y0 + tile_size
        y2 = image.shape[0] - y0
        if x2 >= tile_size and y2 >= tile_size:
            return image[y0:y1, x0:x1]
        # seamless axis wrapping
        tile = np.empty((tile_size, tile_size, image.shape[2]), dtype=self.image.dtype)
        if x2 < tile_size:
            if y2 < tile_size:
                # wrap bottom/right to top/left
                tile[:y2, :x2] = image[y0:, x0:]
                tile[y2:, :x2] = image[:tile_size - y2, x0:]
                tile[:y2, x2:] = image[y0:, :tile_size - x2]
                tile[y2:, x2:] = image[:tile_size - y2, :tile_size - x2]
            else:
                # wrap right to left
                tile[:, :x2] = image[y0:y1, x0:]
                tile[:, x2:] = image[y0:y1, :tile_size - x2]
        else:
            # wrap bottom to top
            tile[:y2] = image[y0:, x0:x1]
            tile[y2:] = image[:tile_size - y2, x0:x1]
        return tile

    def __setitem__(self, key: int | tuple[int, int], tile: NDArray):
        if isinstance(key, int):
            key = self.index_to_xy(key)
        canvas = self.canvas
        scale = self.scale
        tile_size = self.tile_size * scale
        tile_weight = self.tile_weight
        x0 = self.x_tiles[key[0]] * scale
        x1 = x0 + tile_size
        x2 = canvas.shape[1] - x0
        y0 = self.y_tiles[key[1]] * scale
        y1 = y0 + tile_size
        y2 = canvas.shape[0] - y0

        def update(canvas_slice, tile_slice, weight_slice):
            weight_slice = weight_slice.reshape(weight_slice.shape[0], weight_slice.shape[1], 1)
            # undo weighted average, then add new tile with its weights applied and average again
            canvas_slice[:, :, :-1] *= canvas_slice[:, :, -1:]
            canvas_slice[:, :, :-1] += tile_slice * weight_slice
            canvas_slice[:, :, -1:] += weight_slice
            canvas_slice[:, :, :-1] /= canvas_slice[:, :, -1:]

        if x2 >= tile_size and y2 >= tile_size:
            update(canvas[y0:y1, x0:x1], tile, tile_weight)
        elif x2 < tile_size:
            if y2 < tile_size:
                update(canvas[y0:, x0:], tile[:y2, :x2], tile_weight[:y2, :x2])
                update(canvas[:tile_size - y2, x0:], tile[y2:, :x2], tile_weight[y2:, :x2])
                update(canvas[y0:, :tile_size - x2], tile[:y2, x2:], tile_weight[:y2, x2:])
                update(canvas[:tile_size - y2, :tile_size - x2], tile[y2:, x2:], tile_weight[y2:, x2:])
            else:
                update(canvas[y0:y1, x0:], tile[:, :x2], tile_weight[:, :x2])
                update(canvas[y0:y1, :tile_size - x2], tile[:, x2:], tile_weight[:, x2:])
        else:
            update(canvas[y0:, x0:x1], tile[:y2], tile_weight[:y2])
            update(canvas[:tile_size - y2, x0:x1], tile[y2:], tile_weight[y2:])

    def __iter__(self):
        for x in range(len(self.x_tiles)):
            for y in range(len(self.y_tiles)):
                yield (x, y), self[x, y]

    def __len__(self):
        return len(self.x_tiles) * len(self.y_tiles)


def upscale(
    self,
    image: NDArray,
    
    prompt: str,
    steps: int,
    seed: int,
    cfg_scale: float,
    scheduler: Scheduler,
    
    tile_size: int,
    blend: int,
    seamless_axes: SeamlessAxes | str | bool | tuple[bool, bool] | None,

    optimizations: Optimizations,

    step_preview_mode: StepPreviewMode,

    **kwargs
):
    from PIL import Image, ImageOps
    import torch
    import diffusers

    if optimizations.cpu_only:
        device = "cpu"
    else:
        device = self.choose_device()

    pipe = diffusers.StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        revision="fp16" if optimizations.can_use_half(device) else None,
        torch_dtype=torch.float16 if optimizations.can_use_half(device) else torch.float32
    )
    pipe.scheduler = scheduler.create(pipe, None)
    # vae would automatically be made float32 within the pipeline, but it fails to convert after offloading is enabled
    pipe.vae.to(dtype=torch.float32)
    if not optimizations.can_use("sequential_cpu_offload", device):
        pipe = pipe.to(device)
    pipe = optimizations.apply(pipe, device)

    generator = torch.Generator(device="cpu" if device in ("mps", "privateuseone") else device) # MPS and DML do not support the `Generator` API
    if seed is None:
        seed = random.randrange(0, np.iinfo(np.uint32).max)

    if image.shape[2] == 4:
        image = image[:, :, :3]
    tiler = UpscaleTiler(image, 4, tile_size, blend, seamless_axes)
    for i in range(0, len(tiler), optimizations.batch_size):
        batch_size = min(len(tiler)-i, optimizations.batch_size)
        ids = list(range(i, i+batch_size))
        low_res_tiles = [Image.fromarray(tiler[id]) for id in ids]
        high_res_tiles = pipe(
            prompt=[prompt] * batch_size,
            image=low_res_tiles,
            num_inference_steps=steps,
            generator=generator.manual_seed(seed),
            guidance_scale=cfg_scale,
        ).images
        for id, tile in zip(ids, high_res_tiles):
            tiler[id] = np.array(tile)
        step = None
        if step_preview_mode != StepPreviewMode.NONE:
            step = Image.fromarray(tiler.combined().astype(np.uint8))
        yield ImageUpscaleResult(
            (np.asarray(ImageOps.flip(step).convert('RGBA'), dtype=np.float32) / 255.) if step is not None else None,
            i + batch_size,
            len(tiler),
            (i + batch_size) == len(tiler)
        )
    if step_preview_mode == StepPreviewMode.NONE:
        final = Image.fromarray(tiler.combined().astype(np.uint8))
        yield ImageUpscaleResult(
            np.asarray(ImageOps.flip(final).convert('RGBA'), dtype=np.float32) / 255.,
            len(tiler),
            len(tiler),
            True
        )
