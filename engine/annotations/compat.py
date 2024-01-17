import bpy

UNIFORM_COLOR = 'UNIFORM_COLOR' if bpy.app.version >= (3, 4, 0) else '3D_UNIFORM_COLOR'
"""
2D_ and 3D_ prefixed built-in shaders were deprecated in Blender 3.4 and removed in Blender 4.0
"""
