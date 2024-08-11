import importlib.util
import os
import sys
from os import PathLike
from typing import Tuple, Literal, Union, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray, DTypeLike

from .generator_process import RunInSubprocess


"""
This module allows for simple handling of image data in numpy ndarrays in some common formats.

Dimensions:
    2: HW - L
    3: HWC - L/LA/RGB/RGBA
    4: NHWC - batched HWC

Channels:
    1: L
    2: LA
    3: RGB
    4: RGBA
"""


def version_str(version):
    return ".".join(str(x) for x in version)


# find_spec("bpy") will never return None
has_bpy = sys.modules.get("bpy", None) is not None
has_ocio = importlib.util.find_spec("PyOpenColorIO") is not None
has_oiio = importlib.util.find_spec("OpenImageIO") is not None
has_pil = importlib.util.find_spec("PIL") is not None

if has_bpy:
    # frontend
    import bpy
    BLENDER_VERSION = bpy.app.version
    OCIO_CONFIG = os.path.join(bpy.utils.resource_path('LOCAL'), 'datafiles/colormanagement/config.ocio')
    # Easier to share via environment variables than to enforce backends with subprocesses to use their own methods of sharing.
    os.environ["BLENDER_VERSION"] = version_str(BLENDER_VERSION)
    os.environ["BLENDER_OCIO_CONFIG"] = OCIO_CONFIG
else:
    # backend
    BLENDER_VERSION = tuple(int(x) for x in os.environ["BLENDER_VERSION"].split("."))
    OCIO_CONFIG = os.environ["BLENDER_OCIO_CONFIG"]

if TYPE_CHECKING:
    import bpy
    import PIL.Image


def _bpy_version_error(required_version, feature, module):
    if BLENDER_VERSION >= required_version:
        return Exception(f"{module} is unexpectedly missing in Blender {version_str(BLENDER_VERSION)}")
    return Exception(f"{feature} requires Blender {version_str(required_version)} or higher, you are using {version_str(BLENDER_VERSION)}")


def size(array: NDArray) -> Tuple[int, int]:
    if array.ndim == 2:
        return array.shape[1], array.shape[0]
    if array.ndim in [3, 4]:
        return array.shape[-2], array.shape[-3]
    raise ValueError(f"Can't determine size from {array.ndim} dimensions")


def channels(array: NDArray) -> int:
    if array.ndim == 2:
        return 1
    if array.ndim in [3, 4]:
        return array.shape[-1]
    raise ValueError(f"Can't determine channels from {array.ndim} dimensions")


def ensure_alpha(array: NDArray, alpha=None) -> NDArray:
    """
    Args:
        array: Image pixels values.
        alpha: Default alpha value if an alpha channel will be made. Will be inferred from `array.dtype` if None.

    Returns: The converted image or the original image if it already had alpha.
    """
    c = channels(array)
    if c in [2, 4]:
        return array
    if c not in [1, 3]:
        raise ValueError(f"Can't ensure alpha from {c} channels")

    if alpha is None:
        alpha = 0
        if np.issubdtype(array.dtype, np.floating):
            alpha = 1
        elif np.issubdtype(array.dtype, np.integer):
            alpha = np.iinfo(array.dtype).max
    array = ensure_channel_dim(array)
    return np.pad(array, [*[(0, 0)]*(array.ndim-1), (0, 1)], constant_values=alpha)


def ensure_opaque(array: NDArray) -> NDArray:
    """
    Removes the alpha channel if it exists.
    """
    if channels(array) in [2, 4]:
        return array[..., :-1]
    return array


def ensure_channel_dim(array: NDArray) -> NDArray:
    """
    Expands a HW grayscale image to HWC.
    """
    if array.ndim == 2:
        return array[..., np.newaxis]
    return array


def rgb(array: NDArray) -> NDArray:
    """
    Converts a grayscale image to RGB or removes the alpha channel from an RGBA image.
    If the image was already RGB the original array will be returned.
    """
    c = channels(array)
    match channels(array):
        case 1:
            return np.concatenate([ensure_channel_dim(array)] * 3, axis=-1)
        case 2:
            return np.concatenate([array[..., :1]] * 3, axis=-1)
        case 3:
            return array
        case 4:
            return array[..., :3]
    raise ValueError(f"Can't make {c} channels RGB")


def rgba(array: NDArray, alpha=None) -> NDArray:
    """
    Args:
        array: Image pixels values.
        alpha: Default alpha value if an alpha channel will be made. Will be inferred from `array.dtype` if None.

    Returns: The converted image or the original image if it already was RGBA.
    """
    c = channels(array)
    if c == 4:
        return array
    if c == 2:
        l, a = np.split(array, 2, axis=-1)
        return np.concatenate([l, l, l, a], axis=-1)
    return ensure_alpha(rgb(array), alpha)


def grayscale(array: NDArray) -> NDArray:
    """
    Converts `array` into HW or NHWC grayscale. This is intended for converting an
    RGB image that is already visibly grayscale, such as a depth map. It will not
    make a good approximation of perceived lightness of an otherwise colored image.
    """
    if array.ndim == 2:
        return array
    c = channels(array)
    if array.ndim == 3:
        if c in [1, 2]:
            return array[..., 0]
        elif c in [3, 4]:
            return np.max(array[..., :3], axis=-1)
        raise ValueError(f"Can't make {c} channels grayscale")
    elif array.ndim == 4:
        if c in [1, 2]:
            return array[..., :1]
        elif c in [3, 4]:
            return np.max(array[..., :3], axis=-1, keepdims=True)
        raise ValueError(f"Can't make {c} channels grayscale")
    raise ValueError(f"Can't make {array.ndim} dimensions grayscale")


def _passthrough_alpha(from_array, to_array):
    if channels(from_array) not in [2, 4]:
        return to_array
    to_array = np.concatenate([ensure_channel_dim(to_array), from_array[..., -1:]], axis=-1)
    return to_array


def linear_to_srgb(array: NDArray, clamp=True) -> NDArray:
    """
    Args:
        array: Image to convert from linear to sRGB color space. Will be converted to float32 if it isn't already a float dtype.
        clamp: whether to restrict the result between 0..1
    """
    if not np.issubdtype(array.dtype, np.floating):
        array = to_dtype(array, np.float32)
    srgb = ensure_opaque(array)
    srgb = np.where(
        srgb <= 0.0031308,
        srgb * 12.92,
        (np.abs(srgb) ** (1/2.4) * 1.055) - 0.055
        # abs() to suppress `RuntimeWarning: invalid value encountered in power` for negative values
    )
    if clamp:
        # conversion may produce values outside standard range, usually >1
        srgb = np.clip(srgb, 0, 1)
    srgb = _passthrough_alpha(array, srgb)
    return srgb


def srgb_to_linear(array: NDArray) -> NDArray:
    """
    Converts from sRGB to linear color space. Will be converted to float32 if it isn't already a float dtype.
    """
    if not np.issubdtype(array.dtype, np.floating):
        array = to_dtype(array, np.float32)
    linear = ensure_opaque(array)
    linear = np.where(
        linear <= 0.04045,
        linear / 12.92,
        ((linear + 0.055) / 1.055) ** 2.4
    )
    linear = _passthrough_alpha(array, linear)
    return linear


@RunInSubprocess.when_raised
def color_transform(array: NDArray, from_color_space: str, to_color_space: str, *, clamp_srgb=True) -> NDArray:
    """
    Args:
        array: Pixel values in `from_color_space`
        from_color_space: Color space of `array`
        to_color_space: Desired color space
        clamp_srgb: Restrict values inside the standard range when converting to sRGB.

    Returns: Pixel values in `to_color_space`. The image will be converted to RGB/RGBA float32 for most transforms.
        Transforms between linear and sRGB may remain grayscale and keep the original DType if it was floating point.
    """
    # Blender handles Raw and Non-Color images as if they were in Linear color space.
    if from_color_space in ["Raw", "Non-Color"]:
        from_color_space = "Linear"
    if to_color_space in ["Raw", "Non-Color"]:
        to_color_space = "Linear"

    if from_color_space == to_color_space:
        return array
    elif from_color_space == "Linear" and to_color_space == "sRGB":
        return linear_to_srgb(array, clamp_srgb)
    elif from_color_space == "sRGB" and to_color_space == "Linear":
        return srgb_to_linear(array)

    if not has_ocio:
        raise RunInSubprocess

    import PyOpenColorIO as OCIO
    config = OCIO.Config.CreateFromFile(OCIO_CONFIG)
    proc = config.getProcessor(from_color_space, to_color_space).getDefaultCPUProcessor()
    # OCIO requires RGB/RGBA float32.
    # There is a channel agnostic apply(), but I can't seem to get it to work.
    # getOptimizedCPUProcessor() can handle different precisions, but I doubt it would have meaningful use.
    array = to_dtype(array, np.float32)
    c = channels(array)
    if c in [1, 3]:
        array = rgb(array)
        proc.applyRGB(array)
        if clamp_srgb and to_color_space == "sRGB":
            array = np.clip(array, 0, 1)
        return array
    elif c in [2, 4]:
        array = rgba(array)
        proc.applyRGBA(array)
        if clamp_srgb and to_color_space == "sRGB":
            array = np.clip(array, 0, 1)
        return array
    raise ValueError(f"Can't color transform {c} channels")


# inverse=True is often crashing from EXCEPTION_ACCESS_VIOLATION while on frontend.
# Normally this is caused by not running on the main thread or accessing a deleted
# object, neither seem to be the issue here. Doesn't matter if the backend imports
# its own OCIO or the one packaged with Blender.
# Stack trace:
# OpenColorIO_2_2.dll :0x00007FFDE8961160  OpenColorIO_v2_2::GradingTone::validate
# OpenColorIO_2_2.dll :0x00007FFDE8A2BD40  OpenColorIO_v2_2::Processor::isNoOp
# OpenColorIO_2_2.dll :0x00007FFDE882EA00  OpenColorIO_v2_2::CPUProcessor::apply
# PyOpenColorIO.pyd   :0x00007FFDEB0F0E40  pybind11::error_already_set::what
# PyOpenColorIO.pyd   :0x00007FFDEB0F0E40  pybind11::error_already_set::what
# PyOpenColorIO.pyd   :0x00007FFDEB0F0E40  pybind11::error_already_set::what
# PyOpenColorIO.pyd   :0x00007FFDEB0E7510  pybind11::error_already_set::discard_as_unraisable
@RunInSubprocess.when(lambda *_, inverse=False, **__: inverse or not has_ocio)
def render_color_transform(
    array: NDArray,
    exposure: float,
    gamma: float,
    view_transform: str,
    display_device: str,
    look: str,
    *,
    inverse: bool = False,
    color_space: str | None = None,
    clamp_srgb: bool = True,
) -> NDArray:
    import PyOpenColorIO as OCIO

    ocio_config = OCIO.Config.CreateFromFile(OCIO_CONFIG)

    # A reimplementation of `OCIOImpl::createDisplayProcessor` from the Blender source.
    # https://github.com/blender/blender/blob/3816fcd8611bc2836ee8b2a5225b378a02141ce4/intern/opencolorio/ocio_impl.cc#L666
    # Modified to support a final color space transform.
    def create_display_processor(
        config,
        input_colorspace,
        view,
        display,
        look,
        scale,  # Exposure
        exponent,  # Gamma
        inverse,
        color_space
    ):
        group = OCIO.GroupTransform()

        # Exposure
        if scale != 1:
            # Always apply exposure in scene linear.
            color_space_transform = OCIO.ColorSpaceTransform()
            color_space_transform.setSrc(input_colorspace)
            color_space_transform.setDst(OCIO.ROLE_SCENE_LINEAR)
            group.appendTransform(color_space_transform)

            # Make further transforms aware of the color space change
            input_colorspace = OCIO.ROLE_SCENE_LINEAR

            # Apply scale
            matrix_transform = OCIO.MatrixTransform(
                [scale, 0.0, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 0.0, 1.0])
            group.appendTransform(matrix_transform)

        # Add look transform
        use_look = look is not None and len(look) > 0
        if use_look:
            look_output = config.getLook(look).getProcessSpace()
            if look_output is not None and len(look_output) > 0:
                look_transform = OCIO.LookTransform()
                look_transform.setSrc(input_colorspace)
                look_transform.setDst(look_output)
                look_transform.setLooks(look)
                group.appendTransform(look_transform)
                # Make further transforms aware of the color space change.
                input_colorspace = look_output
            else:
                # For empty looks, no output color space is returned.
                use_look = False

        # Add view and display transform
        display_view_transform = OCIO.DisplayViewTransform()
        display_view_transform.setSrc(input_colorspace)
        display_view_transform.setLooksBypass(use_look)
        display_view_transform.setView(view)
        display_view_transform.setDisplay(display)
        group.appendTransform(display_view_transform)

        if color_space is not None:
            group.appendTransform(OCIO.ColorSpaceTransform(input_colorspace if display == "None" else display, color_space))

        # Gamma
        if exponent != 1:
            exponent_transform = OCIO.ExponentTransform([exponent, exponent, exponent, 1.0])
            group.appendTransform(exponent_transform)

        if inverse:
            group.setDirection(OCIO.TransformDirection.TRANSFORM_DIR_INVERSE)

        # Create processor from transform. This is the moment were OCIO validates
        # the entire transform, no need to check for the validity of inputs above.
        return config.getProcessor(group)

    # Exposure and gamma transformations derived from Blender source:
    # https://github.com/blender/blender/blob/3816fcd8611bc2836ee8b2a5225b378a02141ce4/source/blender/imbuf/intern/colormanagement.cc#L867
    scale = 2 ** exposure
    exponent = 1 / max(gamma, np.finfo(np.float32).eps)
    processor = create_display_processor(ocio_config, OCIO.ROLE_SCENE_LINEAR, view_transform, display_device, look if look != 'None' else None, scale, exponent, inverse, color_space)
    array = to_dtype(array, np.float32)
    c = channels(array)
    if c in [1, 3]:
        array = rgb(array)
        processor.getDefaultCPUProcessor().applyRGB(array)
    elif c in [2, 4]:
        array = rgba(array)
        processor.getDefaultCPUProcessor().applyRGBA(array)
    else:
        raise ValueError(f"Can't color transform {c} channels")
    if clamp_srgb and (color_space == "sRGB" or (display_device == "sRGB" and color_space is None)) and not inverse:
        array = np.clip(array, 0, 1)
    return array


def scene_color_transform(array: NDArray, scene: Union["bpy.types.Scene", None] = None, *, inverse: bool = False, color_space: str | None = None, clamp_srgb=True) -> NDArray:
    if scene is None:
        import bpy
        scene = bpy.context.scene
    view = scene.view_settings
    display = scene.display_settings.display_device
    return render_color_transform(
        array,
        view.exposure,
        view.gamma,
        view.view_transform,
        display,
        view.look,
        inverse=inverse,
        clamp_srgb=clamp_srgb,
        color_space=color_space
    )


def _unsigned(dtype: DTypeLike) -> DTypeLike:
    match bits := np.iinfo(dtype).bits:
        case 8:
            return np.uint8
        case 16:
            return np.uint16
        case 32:
            return np.uint32
        case 64:
            return np.uint64
    raise ValueError(f"unexpected bit depth {bits} from {repr(dtype)}")


def to_dtype(array: NDArray, dtype: DTypeLike) -> NDArray:
    """
    Remaps values with respect to ranges rather than simply casting for integer DTypes.
    `integer(0)=float(0)`, `integer.MAX=float(1)`, and signed `integer.MIN+1=float(-1)`
    """
    dtype = np.dtype(dtype)
    from_dtype = array.dtype
    if dtype == from_dtype:
        return array
    from_floating = np.issubdtype(from_dtype, np.floating)
    from_integer = np.issubdtype(from_dtype, np.integer)
    to_floating = np.issubdtype(dtype, np.floating)
    to_integer = np.issubdtype(dtype, np.integer)
    if from_floating and to_floating:
        array = array.astype(dtype)
        if np.finfo(from_dtype).bits > np.finfo(dtype).bits:
            # prevent inf when lowering precision
            array = np.nan_to_num(array)
    elif from_floating and to_integer:
        iinfo = np.iinfo(dtype)
        array = (array.clip(-1 if iinfo.min < 0 else 0, 1) * iinfo.max).round().astype(dtype)
    elif from_integer and to_floating:
        iinfo = np.iinfo(from_dtype)
        array = (array / iinfo.max).astype(dtype)
    elif from_integer and to_integer:
        from_signed = np.issubdtype(from_dtype, np.signedinteger)
        to_signed = np.issubdtype(dtype, np.signedinteger)
        from_bits = np.iinfo(from_dtype).bits
        to_bits = np.iinfo(dtype).bits
        if from_signed:
            from_bits -= 1
        if to_signed:
            to_bits -= 1
        bit_diff = to_bits - from_bits

        if from_signed and not to_signed:
            # unsigned output does not support negative
            array = np.maximum(array, 0)
        if from_signed and to_signed:
            # simpler to handle bit manipulation in unsigned
            sign = np.sign(array)
            array = np.abs(array)

        if bit_diff > 0:
            # Repeat bits rather than using a single left shift
            # so that from_iinfo.max turns into to_iinfo.max
            # and all values remain equally spaced.
            # Example 8 to 16 bits:
            # (incorrect)        0x00FF << 8 = 0xFF00
            # (correct) 0x00FF << 8 | 0x00FF = 0xFFFF
            # Implementation uses multiplication instead of potentially multiple left shifts and ors:
            # 0x00FF * 0x0101 = 0xFFFF
            base = array.astype(_unsigned(dtype))
            m = 0
            for i in range(bit_diff, -1, -from_bits):
                m += 2 ** i
            array = base * m
            remaining_bits = bit_diff % from_bits
            if remaining_bits > 0:
                # when changing between signed and unsigned bit_diff is not a multiple of from_bits
                array |= base >> (from_bits-remaining_bits)
        elif bit_diff < 0:
            array = array.astype(_unsigned(from_dtype), copy=False) >> -bit_diff

        if from_signed and to_signed:
            array = np.multiply(array, sign, dtype=dtype)
        array = array.astype(dtype, copy=False)
    else:
        raise TypeError(f"Unable to convert from {array.dtype} to {dtype}")
    return array


@RunInSubprocess.when(not has_oiio)
def resize(array: NDArray, size: Tuple[int, int], clamp=True):
    no_channels = array.ndim == 2
    if no_channels:
        array = array[..., np.newaxis]
    no_batch = array.ndim < 4
    if no_batch:
        array = array[np.newaxis, ...]
    if clamp:
        c_min = np.min(array, axis=(1, 2), keepdims=True)
        c_max = np.max(array, axis=(1, 2), keepdims=True)

    if has_oiio:
        import OpenImageIO as oiio
        resized = []
        for unbatched in array:
            # OpenImageIO can have batched images, but doesn't support resizing them
            image_in = oiio.ImageBuf(unbatched)
            image_out = oiio.ImageBufAlgo.resize(image_in, roi=oiio.ROI(0, int(size[0]), 0, int(size[1])))
            if image_out.has_error:
                raise Exception(image_out.geterror())
            resized.append(image_out.get_pixels(image_in.spec().format))
        array = np.stack(resized)
    else:
        original_dtype = array.dtype
        if np.issubdtype(original_dtype, np.floating):
            if original_dtype == np.float16:
                # interpolation not implemented for float16 on CPU
                array = to_dtype(array, np.float32)
        elif np.issubdtype(original_dtype, np.integer):
            # integer interpolation only supported for uint8 nearest, nearest-exact or bilinear
            bits = np.iinfo(original_dtype).bits
            array = to_dtype(array, np.float64 if bits >= 32 else np.float32)

        import torch
        array = torch.from_numpy(np.transpose(array, (0, 3, 1, 2)))
        array = torch.nn.functional.interpolate(array, size=(size[1], size[0]), mode="bilinear")
        array = np.transpose(array, (0, 2, 3, 1)).numpy()
        array = to_dtype(array, original_dtype)

    if clamp:
        array = np.clip(array, c_min, c_max)
    if no_batch:
        array = np.squeeze(array, 0)
    if no_channels:
        array = np.squeeze(array, -1)
    return array


def bpy_to_np(image: "bpy.types.Image", *, color_space: str | None = "sRGB", clamp_srgb=True, top_to_bottom=True) -> NDArray:
    """
    Args:
        image: Image to extract pixels values from.
        color_space: The color space to convert to. `None` will apply no color transform.
            Keep in mind that Raw/Non-Color images are handled as if they were in Linear color space.
        clamp_srgb: Restrict values inside the standard range when converting to sRGB.
        top_to_bottom: The y-axis is flipped to a more common standard of `top=0` to `bottom=height-1`.

    Returns: A ndarray copy of `image.pixels` in RGBA float32 format.
    """
    if image.type == "RENDER_RESULT":
        # can't get pixels automatically without rendering again and freezing Blender until it finishes, or saving to disk
        raise ValueError(f"{image.name} image can't be used directly, alternatively use a compositor viewer node")
    array = np.empty((image.size[1], image.size[0], image.channels), dtype=np.float32)
    # foreach_get/set is extremely fast to read/write an entire image compared to alternatives
    # see https://projects.blender.org/blender/blender/commit/9075ec8269e7cb029f4fab6c1289eb2f1ae2858a
    image.pixels.foreach_get(array.ravel())
    if color_space is not None:
        if image.type == "COMPOSITING":
            # Viewer Node
            array = scene_color_transform(array, color_space=color_space, clamp_srgb=clamp_srgb)
        else:
            array = color_transform(array, image.colorspace_settings.name, color_space, clamp_srgb=clamp_srgb)
    if top_to_bottom:
        array = np.flipud(array)
    return rgba(array)


def np_to_bpy(array: NDArray, name=None, existing_image=None, float_buffer=None, color_space: str = "sRGB", top_to_bottom=True) -> "bpy.types.Image":
    """
    Args:
        array: Image pixel values. The y-axis is expected to be ordered `top=0` to `bottom=height-1`.
        name: Name of the image data-block. If None it will be `existing_image.name` or "Untitled".
        existing_image: Image data-block to overwrite.
        float_buffer:
            Make Blender keep data in (`True`) 32-bit float values, or (`False`) 8-bit integer values.
            `None` won't invalidate `existing_image`, but if a new image is created it will be `False`.
        color_space: Color space of `array`.

    Returns: A new Blender image or `existing_image` if it didn't require replacement.
    """
    if array.ndim == 4 and array.shape[0] > 1:
        raise ValueError(f"Can't convert a batched array of {array.shape[0]} images to a Blender image")

    # create or replace image
    import bpy
    width, height = size(array)
    if name is None:
        name = "Untitled" if existing_image is None else existing_image.name
    if existing_image is not None and existing_image.type in ["RENDER_RESULT", "COMPOSITING"]:
        existing_image = None
    elif existing_image is not None and (
            existing_image.size[0] != width
            or existing_image.size[1] != height
            or (existing_image.channels != channels(array) and existing_image.channels != 4)
            or (existing_image.is_float != float_buffer and float_buffer is not None)
    ):
        bpy.data.images.remove(existing_image)
        existing_image = None
    if existing_image is None:
        image = bpy.data.images.new(
            name,
            width=width,
            height=height,
            alpha=channels(array) == 4,
            float_buffer=False if float_buffer is None else float_buffer
        )
    else:
        image = existing_image
        image.name = name
    image.colorspace_settings.name = color_space

    # adjust array pixels to fit into image
    if array.ndim == 4:
        array = array[0]
    if top_to_bottom:
        array = np.flipud(array)
    array = to_dtype(array, np.float32)
    if image.channels == 4:
        array = rgba(array)
    elif image.channels == 3:
        # I believe image.channels only exists for backwards compatibility and modern versions of Blender
        # will always handle images as RGBA. I can't manage to make or import an image and end up with
        # anything but 4 channels. Support for images with 3 channels will be kept just in case.
        array = rgb(array)
    else:
        raise NotImplementedError(f"Blender image unexpectedly has {image.channels} channels")

    # apply pixels to image
    image.pixels.foreach_set(array.ravel())
    image.pack()
    image.update()
    return image


def render_pass_to_np(
    render_pass: "bpy.types.RenderPass",
    size: Tuple[int, int],
    *,
    color_management: bool = False,
    color_space: str | None = None,
    clamp_srgb: bool = True,
    top_to_bottom: bool = True
):
    array = np.empty((*reversed(size), render_pass.channels), dtype=np.float32)
    if BLENDER_VERSION >= (4, 1, 0):
        render_pass.rect.foreach_get(array.reshape(-1))
    else:
        render_pass.rect.foreach_get(array.reshape(-1, render_pass.channels))
    if color_management:
        array = scene_color_transform(array, color_space=color_space, clamp_srgb=clamp_srgb)
    elif color_space is not None:
        array = color_transform(array, "Linear", color_space, clamp_srgb=clamp_srgb)
    if top_to_bottom:
        array = np.flipud(array)
    return array


def np_to_render_pass(
    array: NDArray,
    render_pass: "bpy.types.RenderPass",
    *,
    inverse_color_management: bool = False,
    color_space: str | None = None,
    dtype: DTypeLike = np.float32,
    top_to_bottom: bool = True
):
    if inverse_color_management:
        array = scene_color_transform(array, inverse=True, color_space=color_space)
    elif color_space is not None:
        array = color_transform(color_space, "Linear")
    if channels(array) != render_pass.channels:
        match render_pass.channels:
            case 1:
                array = grayscale(array)
            case 3:
                array = rgb(array)
            case 4:
                array = rgba(array)
            case _:
                raise NotImplementedError(f"Render pass {render_pass.name} unexpectedly requires {render_pass.channels} channels")
    if dtype is not None:
        array = to_dtype(array, dtype)
    if top_to_bottom:
        array = np.flipud(array)
    if BLENDER_VERSION >= (4, 1, 0):
        render_pass.rect.foreach_set(array.reshape(-1))
    else:
        render_pass.rect.foreach_set(array.reshape(-1, render_pass.channels))


def _mode(array, mode):
    if mode is None:
        return array
    elif mode == "RGBA":
        return rgba(array)
    elif mode == "RGB":
        return rgb(array)
    elif mode == "L":
        return grayscale(array)
    elif mode == "LA":
        return ensure_alpha(_passthrough_alpha(array, grayscale(array)))
    raise ValueError(f"mode expected one of {['RGB', 'RGBA', 'L', 'LA', None]}, got {repr(mode)}")


def pil_to_np(image, *, dtype: DTypeLike | None = np.float32, mode: Literal["RGB", "RGBA", "L", "LA"] | None = None) -> NDArray:
    # some modes don't require being converted to RGBA for proper handling in other module functions
    # see for other modes https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    if image.mode not in ["RGB", "RGBA", "L", "LA", "I", "F", "I;16"]:
        image = image.convert("RGBA")
    array = np.array(image)
    if dtype is not None:
        array = to_dtype(array, dtype)
    array = _mode(array, mode)
    return array


def np_to_pil(array: NDArray, *, mode: Literal["RGB", "RGBA", "L", "LA"] | None = None):
    from PIL import Image
    array = to_dtype(array, np.uint8)
    if mode is None:
        if channels(array) == 1 and array.ndim == 3:
            # PIL L mode can't have a channel dimension
            array = array[..., 1]
    else:
        array = _mode(array, mode)
    # PIL does support higher precision modes for a single channel, but I don't see a need for supporting them yet.
    # uint16="I;16", int32="I", float32="F"
    return Image.fromarray(array, mode=mode)


def _dtype_to_type_desc(dtype):
    import OpenImageIO as oiio
    dtype = np.dtype(dtype)
    match dtype:
        case np.uint8:
            return oiio.TypeUInt8
        case np.uint16:
            return oiio.TypeUInt16
        case np.uint32:
            return oiio.TypeUInt32
        case np.uint64:
            return oiio.TypeUInt64
        case np.int8:
            return oiio.TypeInt8
        case np.int16:
            return oiio.TypeInt16
        case np.int32:
            return oiio.TypeInt32
        case np.int64:
            return oiio.TypeInt64
        case np.float16:
            return oiio.TypeHalf
        case np.float32:
            return oiio.TypeFloat
        case np.float64:
            # no oiio.TypeDouble
            return oiio.TypeDesc(oiio.BASETYPE.DOUBLE)
    raise TypeError(f"can't convert {dtype} to OpenImageIO.TypeDesc")


@RunInSubprocess.when(not has_oiio)
def path_to_np(
        path: str | PathLike,
        *,
        dtype: DTypeLike | None = np.float32,
        default_color_space: str | None = None,
        to_color_space: str | None = "sRGB"
) -> NDArray:
    """
    Args:
        path: Path to an image file.
        dtype: Data type of the returned array. `None` won't change the data type. The data type may still change if a color transform occurs.
        default_color_space: The color space that `image_or_path` will be handled as when it can't be determined automatically.
        to_color_space: Color space of the returned array. `None` won't apply a color transform.
    """
    if has_oiio:
        import OpenImageIO as oiio
        image = oiio.ImageInput.open(str(path))
        if image is None:
            raise IOError(oiio.geterror())
        type_desc = image.spec().format
        if dtype is not None:
            type_desc = _dtype_to_type_desc(dtype)
        array = image.read_image(type_desc)
        from_color_space = image.spec().get_string_attribute("oiio:ColorSpace", default_color_space)
        image.close()
    else:
        from PIL import Image
        array = pil_to_np(Image.open(path))
        if dtype is not None:
            array = to_dtype(array, dtype)
        from_color_space = "sRGB"
    if from_color_space is not None and to_color_space is not None:
        array = color_transform(array, from_color_space, to_color_space)
    return array


ImageOrPath = Union[NDArray, "PIL.Image.Image", str, PathLike]
"""Backend compatible image types"""


def image_to_np(
        image_or_path: ImageOrPath | "bpy.types.Image" | None,
        *,
        dtype: DTypeLike | None = np.float32,
        mode: Literal["RGB", "RGBA", "L", "LA"] | None = "RGBA",
        default_color_space: str | None = None,
        to_color_space: str | None = "sRGB",
        size: Tuple[int, int] | None = None,
        top_to_bottom: bool = True
) -> NDArray:
    """
    Opens an image from disk or takes an image object and converts it to `numpy.ndarray`.
    Usable for image argument sanitization when the source can vary in type or format.

    Args:
        image_or_path: Either a file path or an instance of `bpy.types.Image`, `PIL.Image.Image`, or `numpy.ndarray`. `None` will return `None`.
        dtype: Data type of the returned array. `None` won't change the data type. The data type may still change if a color transform occurs.
        mode: Channel mode of the returned array. `None` won't change the mode. The mode may still change if a color transform occurs.
        default_color_space: The color space that `image_or_path` will be handled as when it can't be determined automatically.
        to_color_space: Color space of the returned array. `None` won't apply a color transform.
        size: Resize to specific dimensions. `None` won't change the size.
        top_to_bottom: Flips the image like `bpy_to_np(top_to_bottom=True)` does when `True` and `image_or_path` is a Blender image. Other image sources will only be flipped when `False`.
    """

    if image_or_path is None:
        return None

    # convert image_or_path to numpy.ndarray
    match image_or_path:
        case PathLike() | str():
            array = path_to_np(image_or_path, dtype=dtype, default_color_space=default_color_space, to_color_space=to_color_space)
            from_color_space = None
        case object(__module__="PIL.Image", __class__=type(__name__="Image")):
            # abnormal class check because PIL cannot be imported on frontend
            array = pil_to_np(image_or_path)
            from_color_space = "sRGB"
        case object(__module__="bpy.types", __class__=type(__name__="Image")):
            # abnormal class check because bpy cannot be imported on backend
            array = bpy_to_np(image_or_path, color_space=to_color_space)
            from_color_space = None
        case np.ndarray():
            array = image_or_path
            from_color_space = default_color_space
        case _:
            raise TypeError(f"not an image or path {repr(type(image_or_path))}")

    # apply image requirements
    if not top_to_bottom:
        array = np.flipud(array)
    if from_color_space is not None and to_color_space is not None:
        array = color_transform(array, from_color_space, to_color_space)
    if dtype is not None:
        array = to_dtype(array, dtype)
    array = _mode(array, mode)
    if size is not None:
        array = resize(array, size)

    return array
