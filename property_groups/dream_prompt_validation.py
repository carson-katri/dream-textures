import os
import json
import enum

from ..preferences import StableDiffusionPreferences
from ..generator_process.models import Pipeline, FixItError
from ..generator_process.actions.huggingface_hub import ModelType

def validate(self, context) -> bool:
    match Pipeline[self.pipeline]:
        case Pipeline.STABLE_DIFFUSION:
            if not Pipeline.local_available():
                raise FixItError(
                    "Local generation is not available for the variant of the add-on you have installed. Choose a different Pipeline such as 'DreamStudio'",
                    lambda _, layout: layout.prop(self, "pipeline")
                )
            
            task = ModelType.PROMPT_TO_IMAGE
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

            installed_models = context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences.installed_models
            for model in filter(lambda m: m.model == self.model, installed_models):
                if model.model_type != task.name:
                    raise FixItError(
                        f"""Incorrect model type selected for {task.name.replace('_', ' ').lower()} tasks.
The selected model is for {model.model_type.replace('_', ' ').lower()}.
Select a different model below.""",
                        lambda _, layout: layout.prop(self, "model")
                    )
        case Pipeline.STABILITY_SDK:
            if len(context.preferences.addons[StableDiffusionPreferences.bl_idname].preferences.dream_studio_key) <= 0:
                raise FixItError(
                    f"""No DreamStudio key entered.
Enter your API key below{', or change the pipeline' if Pipeline.local_available() else ''}.""",
                    lambda ctx, layout: layout.prop(ctx.preferences.addons[StableDiffusionPreferences.bl_idname].preferences, "dream_studio_key")
                )
    
    # try:
    #     snapshot_folder = model_snapshot_folder(self.model)
    #     if snapshot_folder is None:
    #         snapshot_folder = self.model
    # except:
    #     snapshot_folder = self.model
    # model_index = os.path.join(snapshot_folder, 'unet', 'config.json')
    # in_channels = 4
    # if os.path.exists(model_index):
    #     with open(model_index) as mi:
    #         in_channels = json.load(mi)['in_channels']
    
    # task = 

    return True