import bpy
from bpy.props import BoolProperty
import sys

def register_section_props():
    for prop in [
        "err_deps_visible",
        "err_wrong_download",
        "err_long_file_path",
        "err_crash",
        "err_silent_fail",
        "err_catchall",
    ]:
        setattr(bpy.types.Scene, prop, BoolProperty(name='', default=False))

def help_section(layout, context):
    def faq_box(is_open, title, steps):
        box = layout.box()
        heading = box.row()
        heading.prop(context.scene, is_open, icon="DOWNARROW_HLT" if getattr(context.scene, is_open) else "RIGHTARROW_THIN", emboss=False, icon_only=True)
        heading.label(text=title)
        if getattr(context.scene, is_open):
            for step in steps:
                if step is not None:
                    box.label(text=step)
        return box
    
    faq_box(
        "err_deps_visible",
        title="Install Dependencies Failed",
        steps=[
            "Open 'Window' > 'Toggle System Console'.",
            "Click 'Install Dependencies' again to repeat the process.",
            "The console window will have a more detailed error.",
            "See below for how to fix the detailed error.",
        ] if sys.platform == 'win32' else [
            "Quit Blender.",
            "Open the system app 'Terminal'",
            "Run the command '/Applications/Blender.app/Contents/MacOS/Blender' to relaunch Blender",
            "Click 'Install Dependencies' again to repeat the process.",
            "The terminal window will have a more detailed error.",
            "See below for how to fix the detailed error.",
        ]
    )

    faq_box(
        "err_wrong_download",
        title="ERROR: 'C:\\...' does not appear to be a Python project: ...",
        steps=[
            "You most likely downloaded the source code, not the bundled addon.",
            "Go to the Releases sidebar tab on GitHub and download the file called 'dream_textures.zip'.",
        ]
    )

    faq_box(
        "err_long_file_path",
        title="ERROR: No matching distribution found for basicsr>=1.4.2 ...",
        steps=[
            "You might have reached Windows' file path character limit.",
            "1. Open up the Window registry (Start > Run > regedit)",
            "2. Navigate to HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem",
            "3. Set LongPathsEnabled to 1",
            "You will need to restart your computer and re-install the dependencies.",
        ]
    )

    faq_box(
        "err_crash",
        title="When I try to generate, Blender hangs and then crashes.",
        steps=[
            "Your computer probably ran out of VRAM. Try reducing the size of the image.",
            "Some GPUs can also work with 'Full Precision' off, which will reduce memory consumption.",
            "If it still fails, open 'Window' > 'Toggle System Console'" if sys.platform == 'win32' else "If it still fails, open the system app 'Terminal'",
            "Run the command '/Applications/Blender.app/Contents/MacOS/Blender' to relaunch Blender" if sys.platform == 'darwin' else None,
            "Try generating an image again.",
            f"The {'console' if sys.platform == 'win32' else 'terminal'} window will have a more detailed error."
        ]
    )

    faq_box(
        "err_silent_fail",
        title="It installs ok, but when I try to generate nothing happens.",
        steps=[
            "Something did go wrong with the install, but it slipped under the radar!",
            "Open 'Window' > 'Toggle System Console'" if sys.platform == 'win32' else "Open the system app 'Terminal'",
            "Run the command '/Applications/Blender.app/Contents/MacOS/Blender' to relaunch Blender" if sys.platform == 'darwin' else None,
            "Try running 'Install Dependencies' again and then generate an image.",
            f"The {'console' if sys.platform == 'win32' else 'terminal'} window will have a more detailed error."
        ]
    )

    faq_box(
        "err_catchall",
        title="It still doesn't work!",
        steps=[
            "Reinstall Blender 3.3 fresh from blender.org then try installing the addon again.",
            "You can completely reset your Blender system by deleting 'C:\\Program Files\\Blender Foundation' and 'C:\\Users\\[YOU]\\AppData\\Roaming\\Blender Foundation'" if sys.platform == 'win32' else None,
            "If you are still unable to fix the issue, please file a bug on GitHub. Include the following:",
            "1. The logs of the error found from the system console",
            "2. Your graphics card specifications.",
            "3. Any steps you already took in an attempt to fix it.",
            "4. A detailed explanation of what happens (or what you expect to happen).",
        ]
    )