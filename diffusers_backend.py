import bpy
from .api import Backend, Model, DreamPrompt, StepCallback, Callback

class DiffusersBackend(Backend):
    name = "HuggingFace Diffusers"
    description = "Local image generation inside of Blender"

    def list_models(self):
        return [
            Model("Stable Diffusion v2.1", "The 2.1 revision of SD", "stabilityai/stable-diffusion-v2-1"),
        ]

    def generate(self, prompt: DreamPrompt, step_callback: StepCallback, callback: Callback):
        pass