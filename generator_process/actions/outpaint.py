from typing import Tuple, Generator
from numpy.typing import NDArray
import numpy as np
from .prompt_to_image import ImageGenerationResult, StepPreviewMode

def outpaint(
    self,

    image: NDArray,

    width: int,
    height: int,

    outpaint_origin: Tuple[int, int],

    **kwargs
) -> Generator[ImageGenerationResult, None, None]:
    from PIL import Image, ImageOps

    init_image = Image.fromarray(image)
    
    if outpaint_origin[0] > init_image.size[0] or outpaint_origin[0] < -width:
        raise ValueError(f"Outpaint origin X ({outpaint_origin[0]}) must be between {-width} and {init_image.size[0]}")
    if outpaint_origin[1] > init_image.size[1] or outpaint_origin[1] < -height:
        raise ValueError(f"Outpaint origin Y ({outpaint_origin[1]}) must be between {-height} and {init_image.size[1]}")
    
    outpaint_bounds = Image.new(
        'RGBA',
        (
            max(init_image.size[0], outpaint_origin[0] + width) - min(0, outpaint_origin[0]),
            max(init_image.size[1], outpaint_origin[1] + height) - min(0, outpaint_origin[1]),
        ),
        (0, 0, 0, 0)
    )
    outpaint_bounds.paste(
        init_image,
        (
            0 if outpaint_origin[0] > 0 else -outpaint_origin[0],
            0 if outpaint_origin[1] > 0 else -outpaint_origin[1],
        )
    )
    offset_origin = (
        max(outpaint_origin[0], 0), # left
        max(outpaint_origin[1], 0), # upper
    )
    # Crop out the area to generate
    inpaint_tile = outpaint_bounds.crop(
        (
            *offset_origin,
            offset_origin[0] + width, # right
            offset_origin[1] + height, # lower
        )
    )

    # tiled batch preview can be saved for another time
    if kwargs['step_preview_mode'] == StepPreviewMode.FAST_BATCH:
        kwargs['step_preview_mode'] = StepPreviewMode.FAST
    elif kwargs['step_preview_mode'] == StepPreviewMode.ACCURATE_BATCH:
        kwargs['step_preview_mode'] = StepPreviewMode.ACCURATE

    def process(step: ImageGenerationResult):
        for i, result_image in enumerate(step.images):
            image = outpaint_bounds.copy()
            image.paste(
                ImageOps.flip(Image.fromarray((result_image.image * 255.).astype(np.uint8))),
                offset_origin
            )
            step.images[i] = image
        return step

    for step in self.inpaint(
        image=np.array(inpaint_tile),
        width=width,
        height=height,
        **kwargs
    ):
        yield process(step)