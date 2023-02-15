import bpy

import numpy as np
from ..generator_process.actions.detect_seamless import SeamlessAxes
from ..generator_process import Generator
from ..preferences import StableDiffusionPreferences


def update(self, context):
    if hasattr(context.area, "regions"):
        for region in context.area.regions:
            if region.type == "UI":
                region.tag_redraw()


class SeamlessResult(bpy.types.PropertyGroup):
    bl_label = "SeamlessResult"
    bl_idname = "dream_textures.SeamlessResult"

    image: bpy.props.PointerProperty(type=bpy.types.Image)
    result: bpy.props.StringProperty(name="Auto-detected", update=update, default=SeamlessAxes.OFF.text)

    def check(self, image):
        if image == self.image or not Generator.shared().can_use():
            return

        if image is not None and (hash_string := image.get('dream_textures_hash', None)) is not None:
            res = None
            def hash_init():
                self.image = image
                self.result = res
            for args in bpy.context.scene.dream_textures_history:
                if args.get('hash', None) == hash_string and args.seamless_axes != SeamlessAxes.AUTO:
                    res = SeamlessAxes(args.seamless_axes).text
                    bpy.app.timers.register(hash_init)
                    return

        can_process = image is not None and image.size[0] >= 8 and image.size[1] >= 8

        def init():
            self.image = image
            self.result = 'Processing' if can_process else SeamlessAxes.OFF.text
        bpy.app.timers.register(init)

        if not can_process:
            return
        pixels = np.empty(image.size[0]*image.size[1]*4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        pixels = pixels.reshape(image.size[1], image.size[0], -1)

        def result(future):
            self.result = future.result().text
        Generator.shared().detect_seamless(pixels).add_done_callback(result)

    def update_args(self, args: dict[str, any], as_id=False):
        if args['seamless_axes'] == SeamlessAxes.AUTO and self.result != 'Processing':
            if as_id:
                args['seamless_axes'] = SeamlessAxes(self.result).id
            else:
                args['seamless_axes'] = SeamlessAxes(self.result)
