from .operators.install_dependencies import InstallDependencies
from .operators.open_latest_version import OpenLatestVersion
from .operators.dream_texture import DreamTexture, ReleaseGenerator
from .operators.view_history import RecallHistoryEntry, SCENE_UL_HistoryList
from .operators.inpaint_area_brush import InpaintAreaStroke
from .operators.upscale import Upscale
from .property_groups.dream_prompt import DreamPrompt
from .ui.panels import dream_texture, history, upscaling
from .preferences import OpenGitDownloads, OpenHuggingFace, OpenWeightsDirectory, OpenRustInstaller, ValidateInstallation, StableDiffusionPreferences

CLASSES = (
    DreamTexture,
    ReleaseGenerator,
    OpenLatestVersion,
    RecallHistoryEntry,
    InpaintAreaStroke,
    Upscale,
    SCENE_UL_HistoryList,
    
    # The order these are registered in matters
    *dream_texture.dream_texture_panels(),
    *upscaling.upscaling_panels(),
    *history.history_panels(),

    upscaling.OpenRealESRGANDownload,
    upscaling.OpenRealESRGANWeightsDirectory
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
