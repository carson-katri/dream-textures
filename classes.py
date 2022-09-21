from .operators.install_dependencies import InstallDependencies
from .operators.open_latest_version import OpenLatestVersion
from .operators.dream_texture import DreamTexture
from .operators.help_panel import HelpPanel
from .operators.view_history import RecallHistoryEntry, SCENE_UL_HistoryList, ViewHistory
from .operators.inpaint_area_brush import InpaintAreaStroke
from .property_groups.dream_prompt import DreamPrompt

from .preferences import OpenGitDownloads, OpenHuggingFace, OpenWeightsDirectory, OpenRustInstaller, ValidateInstallation, StableDiffusionPreferences

CLASSES = (
    DreamTexture,
    HelpPanel,
    OpenLatestVersion,
    ViewHistory,
    RecallHistoryEntry,
    InpaintAreaStroke,
    SCENE_UL_HistoryList,
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
