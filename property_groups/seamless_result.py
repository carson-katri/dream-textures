import bpy

import numpy as np
from ..generator_process import Generator


def update(self, context):
    if hasattr(context.area, "regions"):
        for region in context.area.regions:
            if region.type == "UI":
                region.tag_redraw()


class SeamlessResult(bpy.types.PropertyGroup):
    bl_label = "SeamlessResult"
    bl_idname = "dream_textures.SeamlessResult"

    image: bpy.props.PointerProperty(type=bpy.types.Image)
    result: bpy.props.StringProperty(update=update, default='Off')

    def check(self, image):
        if image == self.image:
            return

        def init():
            self.image = image
            self.result = 'Off' if image is None else 'Processing'
        bpy.app.timers.register(init)

        if image is None or image.size[0] == 0 or image.size[1] == 0:
            return
        pixels = np.empty(image.size[0]*image.size[1]*4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        pixels = pixels.reshape(image.size[1], image.size[0], -1)

        def result(future):
            x, y = future.result()
            def assign():
                self.result = (('X' if x else '') + ('Y' if y else '')) or 'Off'
            bpy.app.timers.register(assign)
        Generator.shared().detect_seamless(pixels).add_done_callback(result)

    def get_axes(self, axes):
        if axes != 'auto':
            return axes
        match self.result:
            case 'Off' | 'Processing':
                return ''
            case default:
                return default.lower()
