from .operators.install_dependencies import InstallDependencies
from .operators.open_latest_version import OpenLatestVersion
from .operators.dream_texture import DreamTexture
from .operators.help_panel import HelpPanel

from .preferences import OpenGitDownloads, OpenHuggingFace, OpenWeightsDirectory, OpenRustInstaller, ValidateInstallation, StableDiffusionPreferences
from .shader_menu import ShaderMenu

CLASSES = (
    DreamTexture,
    HelpPanel,
    OpenLatestVersion,
    ShaderMenu,
)

PREFERENCE_CLASSES = (OpenGitDownloads,
                      InstallDependencies,
                      OpenHuggingFace,
                      OpenWeightsDirectory,
                      OpenRustInstaller,
                      ValidateInstallation,
                      StableDiffusionPreferences)
