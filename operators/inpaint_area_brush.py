import bpy
import math

is_painting = False
last_stroke = None
time = 0
paint_event = None

class InpaintAreaStroke(bpy.types.Operator):
    bl_idname = 'dream_textures.inpaint_area_stroke'
    bl_label = "Mark Area for Inpainting"
    bl_description = "Marks the area for inpainting by setting the alpha to 0 while keeping RGB values"
    bl_options = {'REGISTER'}

    firing_mode: bpy.props.IntProperty()
    alpha_mode: bpy.props.BoolProperty(default=False)

    def invoke(self, context, event):
        global is_painting
        global last_stroke
        global time
        if self.firing_mode == 0: # PRESS
            is_painting = True
        elif self.firing_mode == 2: # RELEASE
            is_painting = False
            last_stroke = None
            time = 0
        elif self.firing_mode == 1 and is_painting: # MOVE
            original_tool = context.workspace.tools.from_space_image_mode('PAINT').idname
        
            bpy.ops.paint.brush_select(image_tool='DRAW', toggle=False)
            brush = bpy.data.brushes["TexDraw"]
            brush.use_pressure_strength = False
            brush.blend = 'ADD_ALPHA' if self.alpha_mode else 'ERASE_ALPHA'
            context.tool_settings.image_paint.brush = brush
            stroke = {
                "name": "stroke",
                "mouse": (event.mouse_x, event.mouse_y - context.scene.tool_settings.unified_paint_settings.size),
                "mouse_event": (0,0),
                "pen_flip" : True,
                "is_start": True if last_stroke is None else False,
                "location": (0, 0, 0),
                "size": context.scene.tool_settings.unified_paint_settings.size,
                "pressure": 1,
                "x_tilt": 0,
                "y_tilt": 0,
                "time": float(time)
            }
            if last_stroke is not None:
                with context.temp_override():
                    bpy.ops.paint.image_paint(stroke=[last_stroke, stroke], mode='NORMAL')
            last_stroke = stroke
            time += 1
            bpy.ops.wm.tool_set_by_id(name=original_tool)
        
        return {"FINISHED"}

    def execute(self, context):
        return {"FINISHED"}

class InpaintAreaBrush(bpy.types.WorkSpaceTool):
    bl_space_type = 'IMAGE_EDITOR'
    bl_context_mode = 'PAINT'

    bl_idname = "dream_textures.inpaint_area_brush"
    bl_label = "Mark Inpaint Area"
    bl_description = "Mark an area for inpainting"
    bl_icon = "brush.gpencil_draw.tint"
    bl_widget = None
    bl_keymap = (
        # Subtract
        (InpaintAreaStroke.bl_idname, {"type": 'LEFTMOUSE', "value": 'PRESS'}, {"properties": [("firing_mode", 0)]}),
        (InpaintAreaStroke.bl_idname, {"type": 'MOUSEMOVE', "value": 'ANY'}, {"properties": [("firing_mode", 1)]}),
        (InpaintAreaStroke.bl_idname, {"type": 'LEFTMOUSE', "value": 'RELEASE'}, {"properties": [("firing_mode", 2)]}),
        # Add - FIXME: Support adding back alpha. So far I have been unable to find a way to customize the custom tool behavior as a brush.
        (InpaintAreaStroke.bl_idname, {"type": 'LEFTMOUSE', "value": 'PRESS', "ctrl": True}, {"properties": [("firing_mode", 0), ("alpha_mode", True)]}),
        (InpaintAreaStroke.bl_idname, {"type": 'MOUSEMOVE', "value": 'ANY', "ctrl": True}, {"properties": [("firing_mode", 1), ("alpha_mode", True)]}),
        (InpaintAreaStroke.bl_idname, {"type": 'LEFTMOUSE', "value": 'RELEASE', "ctrl": True}, {"properties": [("firing_mode", 2), ("alpha_mode", True)]}),
    )

    def draw_settings(self, layout, tool):
        # context.scene.tool_settings.unified_paint_settings
        layout.prop(bpy.context.scene.tool_settings.unified_paint_settings, 'size')