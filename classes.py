from .operators.install_dependencies import InstallDependencies
from .operators.open_latest_version import OpenLatestVersion
from .operators.dream_texture import DreamTexture, ReleaseGenerator, HeadlessDreamTexture, CancelGenerator
from .operators.view_history import SCENE_UL_HistoryList, RecallHistoryEntry, ClearHistory, RemoveHistorySelection, ExportHistorySelection, ImportPromptFile
from .operators.inpaint_area_brush import InpaintAreaBrushActivated
from .operators.upscale import Upscale
from .property_groups.dream_prompt import DreamPrompt
from .ui.panels import dream_texture, history, upscaling, render_properties
from .preferences import OpenHuggingFace, OpenWeightsDirectory, OpenContributors, StableDiffusionPreferences, OpenDreamStudio

from .ui.presets import DREAM_PT_AdvancedPresets, DREAM_MT_AdvancedPresets, AddAdvancedPreset, RestoreDefaultPresets

CLASSES = (
    HeadlessDreamTexture,
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

    DREAM_PT_AdvancedPresets,
    DREAM_MT_AdvancedPresets,
    AddAdvancedPreset,
    
    # The order these are registered in matters
    *dream_texture.dream_texture_panels(),
    *upscaling.upscaling_panels(),
    *history.history_panels(),

    upscaling.OpenRealESRGANDownload,
    upscaling.OpenRealESRGANWeightsDirectory,

    dream_texture.OpenClipSegDownload,
    dream_texture.OpenClipSegWeightsDirectory,
)

PREFERENCE_CLASSES = (
                      DreamPrompt,
                      InstallDependencies,
                      OpenHuggingFace,
                      OpenWeightsDirectory,
                      OpenContributors,
                      RestoreDefaultPresets,
                      OpenDreamStudio,
                      StableDiffusionPreferences)