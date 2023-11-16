from typing import Tuple, Generator
import numpy as np
from ..future import Future
from ...api.models.generation_result import GenerationResult
from ...image_utils import image_to_np, rgba, ImageOrPath

def outpaint(
    self,

    image: ImageOrPath,

    width: int | None,
    height: int | None,

    outpaint_origin: Tuple[int, int],

    **kwargs
) -> Generator[Future, None, None]:

    future = Future()
    yield future

    width = width or 512
    height = height or 512
    image = image_to_np(image)
    
    if outpaint_origin[0] > image.shape[1] or outpaint_origin[0] < -width:
        raise ValueError(f"Outpaint origin X ({outpaint_origin[0]}) must be between {-width} and {image.shape[1]}")
    if outpaint_origin[1] > image.shape[0] or outpaint_origin[1] < -height:
        raise ValueError(f"Outpaint origin Y ({outpaint_origin[1]}) must be between {-height} and {image.shape[0]}")

    outpaint_bounds = np.zeros((
        max(image.shape[0], outpaint_origin[1] + height) - min(0, outpaint_origin[1]),
        max(image.shape[1], outpaint_origin[0] + width) - min(0, outpaint_origin[0]),
        4
    ), dtype=np.float32)

    def paste(under, over, offset):
        under[offset[0]:offset[0] + over.shape[0], offset[1]:offset[1] + over.shape[1]] = over
        return under

    paste(outpaint_bounds, image, (
        0 if outpaint_origin[1] > 0 else -outpaint_origin[1],
        0 if outpaint_origin[0] > 0 else -outpaint_origin[0]
    ))

    offset_origin = (
        max(outpaint_origin[1], 0), # upper
        max(outpaint_origin[0], 0), # left
    )
    # Crop out the area to generate
    inpaint_tile = outpaint_bounds[offset_origin[0]:offset_origin[0]+height, offset_origin[1]:offset_origin[1]+width]

    def process(_, step: [GenerationResult]):
        for res in step:
            res.image = paste(outpaint_bounds.copy(), rgba(res.image), offset_origin)
        future.add_response(step)

    inpaint_generator = self.inpaint(
        image=inpaint_tile,
        width=width,
        height=height,
        **kwargs
    )
    inpaint_future = next(inpaint_generator)
    inpaint_future.check_cancelled = future.check_cancelled
    inpaint_future.add_response_callback(process)
    inpaint_future.add_exception_callback(future.set_exception)
    for _ in inpaint_generator:
        pass

    future.set_done()
