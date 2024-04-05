import numpy as np
from numpy.typing import NDArray
from ..models.optimizations import Optimizations
from ...image_utils import np_to_pil

def controlnet_aux(
    self,

    processor_id: str,
    image: NDArray,

    optimizations: Optimizations,

    **kwargs
) -> NDArray:
    if processor_id == "none":
        return image
    
    from controlnet_aux.processor import Processor
    processor = Processor(processor_id)
    device = self.choose_device(optimizations)
    try:
        processor.processor.to(device)
    except:
        # not all processors can run on the GPU
        pass
    
    processed_image = processor(np_to_pil(image))
    return np.array(processed_image) / 255.0