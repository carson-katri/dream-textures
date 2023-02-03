from .operators.install_dependencies import InstallDependencies, UninstallDependencies
from .operators.open_latest_version import OpenLatestVersion
from .operators.dream_texture import DreamTexture, ReleaseGenerator, CancelGenerator
from .operators.view_history import SCENE_UL_HistoryList, RecallHistoryEntry, ClearHistory, RemoveHistorySelection, ExportHistorySelection, ImportPromptFile
from .operators.inpaint_area_brush import InpaintAreaBrushActivated
from .operators.upscale import Upscale
from .operators.project import ProjectDreamTexture, dream_texture_projection_panels
from .operators.notify_result import NotifyResult
from .property_groups.dream_prompt import DreamPrompt
from .property_groups.seamless_result import SeamlessResult
from .ui.panels import dream_texture, history, upscaling, render_properties
from .preferences import OpenURL, StableDiffusionPreferences, ImportWeights, Model, ModelSearch, InstallModel, PREFERENCES_UL_ModelList

from .ui.presets import DREAM_PT_AdvancedPresets, DREAM_MT_AdvancedPresets, AddAdvancedPreset, RestoreDefaultPresets

CLASSES = (
    *render_properties.render_properties_panels(),
    
    DreamTexture,
    ReleaseGenerator,
    CancelGenerator,
    OpenLatestVersion,
    SCENE_UL_HistoryList,
    RecallHistoryEntry,
    ClearHistory,
    RemoveHistorySelection,
    ExportHistorySelection,
    ImportPromptFile,
    InpaintAreaBrushActivated,
    Upscale,
    ProjectDreamTexture,

    DREAM_PT_AdvancedPresets,
    DREAM_MT_AdvancedPresets,
    AddAdvancedPreset,

    NotifyResult,
    
    # The order these are registered in matters
    *dream_texture.dream_texture_panels(),
    *upscaling.upscaling_panels(),
    *history.history_panels(),
    *dream_texture_projection_panels(),
)

PREFERENCE_CLASSES = (
                      PREFERENCES_UL_ModelList,
                      ModelSearch,
                      InstallModel,
                      Model,
                      DreamPrompt,
                      SeamlessResult,
                      UninstallDependencies,
                      InstallDependencies,
                      OpenURL,
                      ImportWeights,
                      RestoreDefaultPresets,
                      StableDiffusionPreferences)