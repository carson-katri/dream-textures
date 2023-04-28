bl_info = {
    "name": "Test Backend",
    "blender": (3, 1, 0),
    "category": "Paint",
}

import bpy
from typing import List, Tuple
from dream_textures.api import *

class TestBackend(Backend):
    name = "Test"
    description = "A short description of this backend"

    custom_optimization: bpy.props.BoolProperty(name="My Custom Optimization")

    def list_models(self, context) -> List[Model]:
        return []

    def list_schedulers(self, context) -> List[str]:
        return []

    def generate(self, task: Task, model: Model, prompt: Prompt, size: Tuple[int, int] | None, seed: int, steps: int, guidance_scale: float, scheduler: str, seamless_axes: SeamlessAxes, step_preview_mode: StepPreviewMode, iterations: int, step_callback: StepCallback, callback: Callback):
        raise NotImplementedError()

    def draw_speed_optimizations(self, layout, context):
        layout.prop(self, "custom_optimization")

def register():
    bpy.utils.register_class(TestBackend)

def unregister():
    bpy.utils.unregister_class(TestBackend)
