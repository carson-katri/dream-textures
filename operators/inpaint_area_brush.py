import bpy

reset_blend_mode = 'MIX'
reset_curve_preset = 'CUSTOM'
reset_strength = 1.0
class InpaintAreaBrushActivated(bpy.types.GizmoGroup):
    bl_idname = "dream_textures.inpaint_area_brush_activated"
    bl_label = "Inpaint Area Brush Activated"
    bl_space_type = 'IMAGE_EDITOR'
    bl_context_mode = 'PAINT'
    bl_region_type = 'WINDOW'

    def setup(self, context):
        global reset_blend_mode
        global reset_curve_preset
        global reset_strength
        reset_blend_mode = bpy.data.brushes["TexDraw"].blend
        reset_curve_preset = bpy.data.brushes["TexDraw"].curve_preset
        reset_strength = bpy.data.brushes["TexDraw"].strength
        def set_blend():
            bpy.data.brushes["TexDraw"].blend = "ERASE_ALPHA"
            bpy.data.brushes["TexDraw"].curve_preset = "CONSTANT"
            bpy.data.brushes["TexDraw"].strength = 1.0
            bpy.ops.paint.brush_select(image_tool='DRAW', toggle=False)
        bpy.app.timers.register(set_blend)

    def __del__(self):
        bpy.data.brushes["TexDraw"].blend = reset_blend_mode
        bpy.data.brushes["TexDraw"].curve_preset = reset_curve_preset
        bpy.data.brushes["TexDraw"].strength = reset_strength

class InpaintAreaBrush(bpy.types.WorkSpaceTool):
    bl_space_type = 'IMAGE_EDITOR'
    bl_context_mode = 'PAINT'

    bl_idname = "dream_textures.inpaint_area_brush"
    bl_label = "Mark Inpaint Area"
    bl_description = "Mark an area for inpainting"
    bl_icon = "brush.gpencil_draw.tint"
    bl_widget = InpaintAreaBrushActivated.bl_idname

    def draw_settings(self, layout, tool):
        layout.prop(bpy.context.scene.tool_settings.unified_paint_settings, 'size')