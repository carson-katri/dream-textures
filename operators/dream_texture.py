import bpy
import asyncio
import os

from ..preferences import StableDiffusionPreferences
from ..async_loop import *
from ..pil_to_image import *
from ..prompt_engineering import *
from ..absolute_path import WEIGHTS_PATH, absolute_path
from .install_dependencies import are_dependencies_installed

# A shared `Generate` instance.
# This allows the slow model loading process to happen once,
# and re-use the model on subsequent calls.
generator = None

class DreamTexture(bpy.types.Operator):
    bl_idname = "shade.dream_texture"
    bl_label = "Dream Texture"
    bl_description = "Generate a texture with AI"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(self, context):
        return True
    
    def invoke(self, context, event):
        weights_installed = os.path.exists(WEIGHTS_PATH)
        if not weights_installed or not are_dependencies_installed():
            self.report({'ERROR'}, "Please complete setup in the preferences window.")
            return {"FINISHED"}
        else:
            return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        
        scene = context.scene
        
        prompt_box = layout.box()
        prompt_box_heading = prompt_box.row()
        prompt_box_heading.label(text="Prompt")
        prompt_box_heading.prop(scene.dream_textures_prompt, "prompt_structure")
        structure = next(x for x in prompt_structures if x.id == scene.dream_textures_prompt.prompt_structure)
        for segment in structure.structure:
            segment_row = prompt_box.row()
            enum_prop = 'prompt_structure_token_' + segment.id + '_enum'
            is_custom = getattr(scene.dream_textures_prompt, enum_prop) == 'custom'
            if is_custom:
                segment_row.prop(scene.dream_textures_prompt, 'prompt_structure_token_' + segment.id)
            segment_row.prop(scene.dream_textures_prompt, enum_prop, icon_only=is_custom)
        
        size_box = layout.box()
        size_box.label(text="Configuration")
        size_box.prop(scene.dream_textures_prompt, "width")
        size_box.prop(scene.dream_textures_prompt, "height")
        size_box.prop(scene.dream_textures_prompt, "seamless")
        
        init_img_box = layout.box()
        init_img_heading = init_img_box.row()
        init_img_heading.prop(scene.dream_textures_prompt, "use_init_img")
        init_img_heading.label(text="Init Image")
        if scene.dream_textures_prompt.use_init_img:
            init_img_box.template_ID(context.scene, "init_img", open="image.open")
            init_img_box.prop(scene.dream_textures_prompt, "strength")
            init_img_box.prop(scene.dream_textures_prompt, "fit")

        advanced_box = layout.box()
        advanced_box_heading = advanced_box.row()
        advanced_box_heading.prop(scene.dream_textures_prompt, "show_advanced", icon="DOWNARROW_HLT" if scene.dream_textures_prompt.show_advanced else "RIGHTARROW_THIN", emboss=False, icon_only=True)
        advanced_box_heading.label(text="Advanced Configuration")
        if scene.dream_textures_prompt.show_advanced:
            advanced_box.prop(scene.dream_textures_prompt, "full_precision")
            advanced_box.prop(scene.dream_textures_prompt, "seed")
            # advanced_box.prop(self, "iterations") # Disabled until supported by the addon.
            advanced_box.prop(scene.dream_textures_prompt, "steps")
            advanced_box.prop(scene.dream_textures_prompt, "cfgscale")
            advanced_box.prop(scene.dream_textures_prompt, "sampler")
            advanced_box.prop(scene.dream_textures_prompt, "show_steps")

    def cancel(self, context):
        pass

    async def dream_texture(self, context):
        history_entry = context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences.history.add()
        for prop in context.scene.dream_textures_prompt.__annotations__.keys():
            if hasattr(history_entry, prop):
                setattr(history_entry, prop, getattr(context.scene.dream_textures_prompt, prop))

        generated_prompt = context.scene.dream_textures_prompt.generate_prompt()

        # Support Apple Silicon GPUs as much as possible.
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        from ..stable_diffusion.ldm.generate import Generate
        from omegaconf import OmegaConf
        
        config  = absolute_path('stable_diffusion/configs/models.yaml')
        model   = 'stable-diffusion-1.4'

        models  = OmegaConf.load(config)
        width   = models[model].width
        height  = models[model].height
        config  = absolute_path('stable_diffusion/' + models[model].config)
        weights = absolute_path('stable_diffusion/' + models[model].weights)

        global generator
        if generator is None:
            generator = Generate(
                width=width,
                height=height,
                sampler_name=context.scene.dream_textures_prompt.sampler,
                weights=weights,
                full_precision=context.scene.dream_textures_prompt.full_precision,
                seamless=context.scene.dream_textures_prompt.seamless,
                config=config,
            )
            generator.load_model()

        node_tree = context.material.node_tree if hasattr(context, 'material') else None
        window_manager = context.window_manager
        screen = context.screen
        last_data_block = None
        scene = context.scene

        def image_writer(image, seed, upscaled=False):
            nonlocal last_data_block
            # Only use the non-upscaled texture, as upscaling is currently unsupported by the addon.
            if not upscaled:
                if last_data_block is not None:
                    bpy.data.images.remove(last_data_block)
                    last_data_block = None
                image = pil_to_image(image, name=f"{seed}")
                if node_tree is not None:
                    nodes = node_tree.nodes
                    texture_node = nodes.new("ShaderNodeTexImage")
                    texture_node.image = image
                    nodes.active = texture_node
                for area in screen.areas:
                    if area.type == 'IMAGE_EDITOR':
                        area.spaces.active.image = image
                window_manager.progress_end()
        
        def view_step(samples, step):
            step_progress(samples, step)
            nonlocal last_data_block
            for area in screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    step_image = pil_to_image(generator._sample_to_image(samples), name=f'Step {step + 1}/{scene.dream_textures_prompt.steps}')
                    area.spaces.active.image = step_image
                    if last_data_block is not None:
                        bpy.data.images.remove(last_data_block)
                    last_data_block = step_image
                    return # Only perform this on the first image editor found.
        
        def step_progress(samples, step):
            window_manager.progress_update(step)

        def perform():
            window_manager.progress_begin(0, scene.dream_textures_prompt.steps)
            generator.prompt2image(
                # prompt string (no default)
                prompt=generated_prompt,
                # iterations (1); image count=iterations
                iterations=scene.dream_textures_prompt.iterations,
                # refinement steps per iteration
                steps=scene.dream_textures_prompt.steps,
                # seed for random number generator
                seed=None if scene.dream_textures_prompt.seed == -1 else scene.dream_textures_prompt.seed,
                # width of image, in multiples of 64 (512)
                width=scene.dream_textures_prompt.width,
                # height of image, in multiples of 64 (512)
                height=scene.dream_textures_prompt.height,
                # how strongly the prompt influences the image (7.5) (must be >1)
                cfg_scale=scene.dream_textures_prompt.cfgscale,
                # path to an initial image - its dimensions override width and height
                init_img=scene.init_img.filepath if scene.init_img is not None and scene.dream_textures_prompt.use_init_img else None,

                fit=scene.dream_textures_prompt.fit,
                # strength for noising/unnoising init_img. 0.0 preserves image exactly, 1.0 replaces it completely
                strength=scene.dream_textures_prompt.strength,
                # strength for GFPGAN. 0.0 preserves image exactly, 1.0 replaces it completely
                gfpgan_strength=0.0, # 0 disables upscaling, which is currently not supported by the addon.
                # image randomness (eta=0.0 means the same seed always produces the same image)
                ddim_eta=0.0,
                # a function or method that will be called each step
                step_callback=view_step if scene.dream_textures_prompt.show_steps else step_progress,
                # a function or method that will be called each time an image is generated
                image_callback=image_writer,
                
                sampler_name=scene.dream_textures_prompt.sampler
            )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, perform)

    def execute(self, context):
        async_task = asyncio.ensure_future(self.dream_texture(context))
        # async_task.add_done_callback(done_callback)
        ensure_async_loop()

        return {'FINISHED'}