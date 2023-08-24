from .actor import Actor

class Generator(Actor):
    """
    The actor used for all background processes.
    """

    from .actions.choose_device import choose_device
    from .actions.load_model import load_model
    from .actions.prompt_to_image import prompt_to_image
    from .actions.image_to_image import image_to_image
    from .actions.inpaint import inpaint
    from .actions.outpaint import outpaint
    from .actions.upscale import upscale
    from .actions.depth_to_image import depth_to_image
    from .actions.control_net import control_net
    from .actions.huggingface_hub import hf_snapshot_download, hf_list_models, hf_list_installed_models, set_checkpoint_links
    from .actions.ocio_transform import ocio_transform
    from .actions.convert_original_stable_diffusion_to_diffusers import convert_original_stable_diffusion_to_diffusers
    from .actions.detect_seamless import detect_seamless
