import numpy as np
from numpy.typing import NDArray
from ..models.optimizations import Optimizations

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
    processor.processor.to(device)
    
    processed_image = processor(image)
    return np.array(processed_image) / 255.0