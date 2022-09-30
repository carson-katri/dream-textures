from .operators.install_dependencies import InstallDependencies
from .operators.open_latest_version import OpenLatestVersion
from .operators.dream_texture import DreamTexture, HeadlessDreamTexture, ReleaseGenerator
from .operators.view_history import RecallHistoryEntry, SCENE_UL_HistoryList
from .operators.inpaint_area_brush import InpaintAreaStroke
from .property_groups.dream_prompt import DreamPrompt
from .ui.panel import panels, history_panels, troubleshooting_panels
from .ui.render_properties import RenderPropertiesPanel
from .preferences import OpenGitDownloads, OpenHuggingFace, OpenWeightsDirectory, OpenRustInstaller, ValidateInstallation, StableDiffusionPreferences

CLASSES = (
    HeadlessDreamTexture,
    
    DreamTexture,
    ReleaseGenerator,
    OpenLatestVersion,
    RecallHistoryEntry,
    
    InpaintAreaStroke,

    SCENE_UL_HistoryList,

    *panels(),
    *history_panels(),
    *troubleshooting_panels(),
    RenderPropertiesPanel,
)

PREFERENCE_CLASSES = (
                      DreamPrompt,
                      OpenGitDownloads,
                      InstallDependencies,
                      OpenHuggingFace,
                      OpenWeightsDirectory,
                      OpenRustInstaller,
                      ValidateInstallation,
                      StableDiffusionPreferences)