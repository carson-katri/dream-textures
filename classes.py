from .operators.install_dependencies import InstallDependencies
from .operators.open_latest_version import OpenLatestVersion
from .operators.dream_texture import DreamTexture
from .operators.help_panel import HelpPanel
from .operators.view_history import RecallHistoryEntry, SCENE_UL_HistoryList, ViewHistory
from .property_groups.dream_prompt import DreamPrompt

from .preferences import OpenGitDownloads, OpenHuggingFace, OpenWeightsDirectory, OpenRustInstaller, ValidateInstallation, StableDiffusionPreferences
from .shader_menu import ShaderMenu

CLASSES = (
    DreamTexture,
    HelpPanel,
    OpenLatestVersion,
    ShaderMenu,
    ViewHistory,
    RecallHistoryEntry,
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
