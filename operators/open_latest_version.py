import requests
import json
import bpy
import webbrowser
from ..version import VERSION, version_tag, version_tuple

REPO_OWNER = "carson-katri"
REPO_NAME = "dream-textures"

latest_version = VERSION
def check_for_updates():
    try:
        global latest_version
        response = requests.get(f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases")
        releases = response.json()
        latest_version = version_tuple(releases[0]['tag_name'])
    except:
        pass

def new_version_available():
    return not latest_version == VERSION
def force_show_update():
    global VERSION
    VERSION = (0,0,0)

class OpenLatestVersion(bpy.types.Operator):
    bl_idname = "stable_diffusion.open_latest_version"
    bl_label = f"Update Available..."
    bl_description = ("Opens a window to download the latest release from GitHub")
    bl_options = {"REGISTER", "INTERNAL"}

    @classmethod
    def poll(self, context):
        return True

    def execute(self, context):
        webbrowser.open(f'https://github.com/carson-katri/dream-textures/releases/tag/{version_tag(latest_version)}')

        return {"FINISHED"}