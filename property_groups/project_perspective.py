import bpy

class ProjectPerspective(bpy.types.PropertyGroup):
    bl_label = "ProjectPerspective"
    bl_idname = "dream_textures.ProjectPerspective"

    name: bpy.props.StringProperty(name="Name", default="Perspective")
    matrix: bpy.props.FloatVectorProperty(name="", size=4*4)