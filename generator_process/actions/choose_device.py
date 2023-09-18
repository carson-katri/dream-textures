import importlib.util
import sys

def choose_device(self, optimizations) -> str:
    """
    Automatically select which PyTorch device to use.
    """
    if optimizations.cpu_only:
        return "cpu"
    
    import torch
    
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    elif importlib.util.find_spec("torch_directml"):
        import torch_directml
        if torch_directml.is_available():
            torch.utils.rename_privateuse1_backend("dml")
            return "dml"
    return "cpu"