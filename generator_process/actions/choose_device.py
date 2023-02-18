from ..models import Pipeline

def choose_device(self) -> str:
    """
    Automatically select which PyTorch device to use.
    """
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    if Pipeline.directml_available():
        import torch_directml
        if torch_directml.is_available():
            # can be named better when torch.utils.rename_privateuse1_backend() is released
            return "privateuseone"
    return "cpu"