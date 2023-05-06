bl_info = {
    "name": "CoreML Backend",
    "blender": (3, 1, 0),
    "category": "Paint",
}

from multiprocessing import current_process
import site
import sys
import os

def _load_dependencies():
    site.addsitedir(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".python_dependencies"))
    deps = sys.path.pop(-1)
    sys.path.insert(0, deps)

if current_process().name == "__actor__":
    _load_dependencies()
else:
    import bpy
    from typing import Tuple
    from dream_textures.api import Backend, Task, Model, Prompt, SeamlessAxes, StepPreviewMode, StepCallback, Callback
    from dream_textures.diffusers_backend import DiffusersBackend
    from .actor import CoreMLActor

    class CoreMLBackend(Backend):
        name = "CoreML"
        description = "CPU/GPU/NE accelerated generation on Apple Silicon"

        compute_unit: bpy.props.EnumProperty(
            name="Compute Unit",
            items=(
                ('ALL', 'All', 'Use all compute units available, including the neural engine'),
                ('CPU_ONLY', 'CPU', 'Limit the model to only use the CPU'),
                ('CPU_AND_GPU', 'CPU and GPU', 'Use both the CPU and GPU, but not the neural engine'),
                ('CPU_AND_NE', 'CPU and NE', 'Use both the CPU and neural engine, but not the GPU'),
            )
        )

        def list_models(self, context):
            return DiffusersBackend.list_models(self, context)
        
        def list_schedulers(self, context):
            return [
                "DDIM",
                "DPM Solver Multistep",
                "Euler Ancestral Discrete",
                "Euler Discrete",
                "LMS Discrete",
                "PNDM"
            ]

        def draw_speed_optimizations(self, layout, context):
            layout.prop(self, "compute_unit")

        def generate(self, task: Task, model: Model, prompt: Prompt, size: Tuple[int, int] | None, seed: int, steps: int, guidance_scale: float, scheduler: str, seamless_axes: SeamlessAxes, step_preview_mode: StepPreviewMode, iterations: int, step_callback: StepCallback, callback: Callback):
            gen: CoreMLActor = CoreMLActor.shared()
            future = gen.generate(
                model=model.id.replace('models--', '').replace('--', '/'),
                prompt=prompt.positive,
                negative_prompt=prompt.negative,
                size=size,
                seed=seed,
                steps=steps,
                guidance_scale=guidance_scale,
                scheduler=scheduler,
                seamless_axes=seamless_axes,
                step_preview_mode=step_preview_mode,
                iterations=iterations,
                compute_unit=self.compute_unit,
                controlnet=None,
                controlnet_inputs=[]
            )
            def on_step(_, result):
                step_callback(result)
            def on_done(future):
                result = future.result(last_only=True)
                callback([result])
            def on_exception(_, exception):
                callback(exception)
            future.add_response_callback(on_step)
            future.add_exception_callback(on_exception)
            future.add_done_callback(on_done)

    def register():
        bpy.utils.register_class(CoreMLBackend)

    def unregister():
        bpy.utils.unregister_class(CoreMLBackend)
