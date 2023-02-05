from ..preferences import StableDiffusionPreferences, _template_model_download_progress, InstallModel
from ..generator_process.models import Pipeline, FixItError
from ..generator_process.actions.huggingface_hub import ModelType
from ..preferences import OpenURL

def validate(self, context, task: ModelType | None = None) -> bool:
    if task is None:
        if self.use_init_img:
            match self.init_img_action:
                case 'modify':
                    match self.modify_action_source_type:
                        case 'color':
                            task = ModelType.PROMPT_TO_IMAGE
                        case 'depth_generated' | 'depth_map' | 'depth':
                            task = ModelType.DEPTH
                case 'inpaint' | 'outpaint':
                    task = ModelType.INPAINTING
        if task is None:
            task = ModelType.PROMPT_TO_IMAGE

    # Check if the pipeline supports the task.
    pipeline = Pipeline[self.pipeline]
    match task:
        case ModelType.DEPTH:
            if not pipeline.depth():
                raise FixItError(
                    f"""The selected pipeline does not support {task.name.replace('_', ' ').lower()} tasks.
Select a different pipeline below.""",
                    lambda _, layout: layout.prop(self, "pipeline")
                )

    # Pipeline-specific checks
    match pipeline:
        case Pipeline.STABLE_DIFFUSION:
            if not Pipeline.local_available():
                raise FixItError(
                    "Local generation is not available for the variant of the add-on you have installed. Choose a different Pipeline such as 'DreamStudio'",
                    lambda _, layout: layout.prop(self, "pipeline")
                )

            installed_models = context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences.installed_models
            model = next((m for m in installed_models if m.model_base == self.model), None)
            if model is None:
                raise FixItError("No model selected.", lambda _, layout: layout.prop(self, "model"))
            else:
                if model.model_type != task.name:
                    def fix_model(context, layout):
                        layout.prop(self, "model")
                        if not any(m.model_type == task.name for m in installed_models):
                            if not _template_model_download_progress(context, layout):
                                layout.label(text="You do not have any compatible models downloaded:")
                                install_model = layout.operator(InstallModel.bl_idname, text=f"Download {task.recommended_model()} (Recommended)", icon="IMPORT")
                                install_model.model = task.recommended_model()
                                install_model.prefer_fp16_revision = context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences.prefer_fp16_revision
                    raise FixItError(
                        f"""Incorrect model type selected for {task.name.replace('_', ' ').lower()} tasks.
The selected model is for {model.model_type.replace('_', ' ').lower()}.
Select a different model below.""",
                        fix_model
                    )
        case Pipeline.STABILITY_SDK:
            if len(context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences.dream_studio_key) <= 0:
                raise FixItError(
                    f"""No DreamStudio key entered.
Enter your API key below{', or change the pipeline' if Pipeline.local_available() else ''}.""",
                    lambda ctx, layout: layout.prop(ctx.preferences.addons[StableDiffusionPreferences.bl_idname].preferences, "dream_studio_key")
                )

    init_image = None
    if self.use_init_img:
        match self.init_img_src:
            case 'file':
                init_image = context.scene.init_img
            case 'open_editor':
                for area in context.screen.areas:
                    if area.type == 'IMAGE_EDITOR':
                        if area.spaces.active.image is not None:
                            init_image = area.spaces.active.image
        if init_image is not None and init_image.type == 'RENDER_RESULT':
            def fix_init_img(ctx, layout):
                layout.prop(self, "init_img_src", expand=True)
                if self.init_img_src == 'file':
                    layout.template_ID(context.scene, "init_img", open="image.open")
                layout.label(text="Or, enable the render pass to generate after each render.")
                layout.operator(OpenURL.bl_idname, text="Learn More", icon="QUESTION").url = "https://github.com/carson-katri/dream-textures/blob/main/docs/RENDER_PASS.md"
            raise FixItError("""'Render Result' cannot be used as a source image.
Save the image then open the file to use it as a source image.""",
                fix_init_img
            )

    return True