from .operators.install_dependencies import InstallDependencies
from .operators.open_latest_version import OpenLatestVersion
from .operators.dream_texture import DreamTexture, ReleaseGenerator
from .operators.help_panel import HelpPanel
from .operators.view_history import RecallHistoryEntry, SCENE_UL_HistoryList, ViewHistory
from .operators.inpaint_area_brush import InpaintAreaStroke
from .property_groups.dream_prompt import DreamPrompt
from .ui.panel import DREAM_PT_dream_panel, DREAM_PT_dream_node_panel
from .preferences import OpenGitDownloads, OpenHuggingFace, OpenWeightsDirectory, OpenRustInstaller, ValidateInstallation, StableDiffusionPreferences

CLASSES = (
    DreamTexture,
    ReleaseGenerator,
    HelpPanel,
    OpenLatestVersion,
    ViewHistory,
    RecallHistoryEntry,
    InpaintAreaStroke,
    SCENE_UL_HistoryList,
    DREAM_PT_dream_panel,
    DREAM_PT_dream_node_panel,
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
