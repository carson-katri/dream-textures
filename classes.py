from .operators.install_dependencies import InstallDependencies
from .operators.open_latest_version import OpenLatestVersion
from .operators.dream_texture import DreamTexture
from .preferences import OpenGitDownloads, OpenHuggingFace, OpenWeightsDirectory, ValidateInstallation, StableDiffusionPreferences
from .shader_menu import ShaderMenu

CLASSES = (
    DreamTexture,
    OpenLatestVersion,
    ShaderMenu,
)

PREFERENCE_CLASSES = (OpenGitDownloads,
                      InstallDependencies,
                      OpenHuggingFace,
                      OpenWeightsDirectory,
                      ValidateInstallation,
                      StableDiffusionPreferences)
