import importlib.util
import os
import sys
from os import PathLike
from typing import Tuple, Literal, Union, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray, DTypeLike


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


def height_width(array: NDArray) -> Tuple[int, int]:
    if array.ndim == 2:
        return array.shape[0:2]
    if array.ndim in [3, 4]:
        return array.shape[-3:-1]
    raise ValueError(f"Can't determine height and width from {array.ndim} dimensions")


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
        raise _bpy_version_error((3, 5, 0), "color transform", "PyOpenColorIO")
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


def resize(array: NDArray, size: Tuple[int, int]):
    # currently only supported on the backend
    # frontend could use OpenImageIO for Blender version >= 3.5.0

    import torch

    no_channels = array.ndim == 2
    if no_channels:
        array = array[..., np.newaxis]
    no_batch = array.ndim < 4
    if no_batch:
        array = array[np.newaxis, ...]

    original_dtype = array.dtype
    if np.issubdtype(original_dtype, np.floating):
        if original_dtype == np.float16:
            # interpolation not implemented for float16 on CPU
            array = to_dtype(array, np.float32)
    elif np.issubdtype(original_dtype, np.integer):
        # integer interpolation only supported for uint8 nearest, nearest-exact or bilinear
        bits = np.iinfo(original_dtype).bits
        array = to_dtype(array, np.float64 if bits >= 32 else np.float32)

    array = torch.from_numpy(np.transpose(array, (0, 3, 1, 2)))
    array = torch.nn.functional.interpolate(array, size=size, mode="bilinear")
    array = np.transpose(np.array(array), (0, 2, 3, 1))

    array = to_dtype(array, original_dtype)

    if no_batch:
        array = np.squeeze(array, 0)
    if no_channels:
        array = np.squeeze(array, -1)
    return array


def bpy_to_np(image: "bpy.types.Image", color_space: str | None = "sRGB") -> NDArray:
    """
    Args:
        image: Image to extract pixels values from.
        color_space: The color space to convert to. `None` will apply no color transform.
            Keep in mind that Raw/Non-Color images are handled as if they were in Linear color space.

    Returns: A ndarray copy of `image.pixels` in RGBA float32 format.
        The y-axis is flipped to a more common standard of `top=0` to `bottom=height-1`.
    """
    array = np.empty((image.size[1], image.size[0], image.channels), dtype=np.float32)
    # foreach_get/set is extremely fast to read/write an entire image compared to alternatives
    # see https://projects.blender.org/blender/blender/commit/9075ec8269e7cb029f4fab6c1289eb2f1ae2858a
    image.pixels.foreach_get(array.ravel())
    # TODO check for image.type == "RENDER_RESULT" or "COMPOSITING"
    if color_space is not None:
        array = color_transform(array, image.colorspace_settings.name, color_space)
    return rgba(np.flipud(array))


def np_to_bpy(array: NDArray, name=None, existing_image=None, float_buffer=None, color_space: str = "sRGB") -> "bpy.types.Image":
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
    height, width = height_width(array)
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
        return _passthrough_alpha(array, grayscale(array))
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
        case np.float32 | np.float64:
            return oiio.TypeFloat
        case np.float64:
            # no oiio.TypeDouble
            return oiio.TypeDesc(oiio.BASETYPE.DOUBLE)
    raise TypeError(f"can't convert {dtype} to OpenImageIO.TypeDesc")


ImageOrPath = Union[NDArray, "PIL.Image.Image", str, PathLike]
"""Backend compatible image types"""


def image_to_np(
        image_or_path: ImageOrPath | "bpy.types.Image" | None,
        *,
        dtype: DTypeLike | None = np.float32,
        mode: Literal["RGB", "RGBA", "L", "LA"] | None = "RGBA",
        color_space: str | None = "sRGB",
        size: Tuple[int, int] | None = None
) -> NDArray:
    """
    Opens an image from disk or takes an image object and converts it to `numpy.ndarray`.
    Usable for image argument sanitization when the source can vary in type or format.

    Args:
        image_or_path: Either a file path or an instance of `bpy.types.Image`, `PIL.Image.Image`, or `numpy.ndarray`. `None` will return `None`.
        dtype: Data type of the returned array. `None` won't change the data type. The data type may still change if a color transform occurs.
        mode: Channel mode of the returned array. `None` won't change the mode. The mode may still change if a color transform occurs.
        color_space: Color space of the returned array. `None` won't apply a color transform. If `image_or_path` is of type `numpy.ndarray` no color transform will occur.
        size: Resize to specific dimensions. `None` won't change the size.
    """

    if image_or_path is None:
        return None
    array = None
    from_color_space = None

    # convert image_or_path to numpy.ndarray
    match image_or_path:
        case PathLike() | str():
            if has_oiio:
                import OpenImageIO as oiio
                image_or_path = oiio.ImageInput.open(str(image_or_path))
                if image_or_path is None:
                    raise IOError(oiio.geterror())
                type_desc = image_or_path.spec().format
                if dtype is not None:
                    type_desc = _dtype_to_type_desc(dtype)
                array = image_or_path.read_image(type_desc)
                from_color_space = image_or_path.spec().get_string_attribute("oiio:ColorSpace", "sRGB")
            elif has_pil:
                from PIL import Image
                array = pil_to_np(Image.open(image_or_path))
                from_color_space = "sRGB"
            else:
                raise _bpy_version_error((3, 5, 0), "read image", "OpenImageIO")
        case object(__module__="PIL.Image", __class__=type(__name__="Image")):
            # abnormal class check because PIL cannot be imported on frontend
            array = pil_to_np(image_or_path)
            from_color_space = "sRGB"
        case object(__module__="bpy.types", __class__=type(__name__="Image")):
            # abnormal class check because bpy cannot be imported on backend
            array = bpy_to_np(image_or_path, color_space)
        case np.ndarray():
            # no way to tell what color space
            array = image_or_path
        case _:
            raise TypeError(f"not an image or path {repr(type(image_or_path))}")

    # apply image requirements
    if from_color_space is not None and color_space is not None:
        array = color_transform(array, from_color_space, color_space)
    if dtype is not None:
        array = to_dtype(array, dtype)
    array = _mode(array, mode)
    if size is not None:
        array = resize(array, size)

    return array