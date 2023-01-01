from .operators.install_dependencies import InstallDependencies
from .operators.open_latest_version import OpenLatestVersion
from .operators.dream_texture import DreamTexture, ReleaseGenerator, CancelGenerator
from .operators.view_history import SCENE_UL_HistoryList, RecallHistoryEntry, ClearHistory, RemoveHistorySelection, ExportHistorySelection, ImportPromptFile
from .operators.inpaint_area_brush import InpaintAreaBrushActivated
from .operators.upscale import Upscale
from .operators.project import ProjectDreamTexture, dream_texture_projection_panels, AddPerspective, RemovePerspective, LoadPerspective, SCENE_UL_ProjectPerspectiveList
from .property_groups.dream_prompt import DreamPrompt
from .property_groups.project_perspective import ProjectPerspective
from .ui.panels import dream_texture, history, upscaling, render_properties
from .preferences import OpenHuggingFace, OpenContributors, StableDiffusionPreferences, OpenDreamStudio, ImportWeights, Model, ModelSearch, InstallModel, PREFERENCES_UL_ModelList

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
    AddPerspective,
    RemovePerspective,
    LoadPerspective,
    SCENE_UL_ProjectPerspectiveList,

    DREAM_PT_AdvancedPresets,
    DREAM_MT_AdvancedPresets,
    AddAdvancedPreset,
    
    # The order these are registered in matters
    *dream_texture.dream_texture_panels(),
    *upscaling.upscaling_panels(),
    *history.history_panels(),
    *dream_texture_projection_panels(),

    dream_texture.OpenClipSegDownload,
    dream_texture.OpenClipSegWeightsDirectory,
)

PREFERENCE_CLASSES = (
                      PREFERENCES_UL_ModelList,
                      ModelSearch,
                      InstallModel,
                      Model,
                      DreamPrompt,
                      ProjectPerspective,
                      InstallDependencies,
                      OpenHuggingFace,
                      ImportWeights,
                      OpenContributors,
                      RestoreDefaultPresets,
                      OpenDreamStudio,
                      StableDiffusionPreferences)