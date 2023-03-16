import bpy
import gpu
from gpu_extras.batch import batch_for_shader
import numpy as np
from ..node import DreamTexturesNode
from ..annotations import openpose

class NodeString(DreamTexturesNode):
    bl_idname = "dream_textures.node_string"
    bl_label = "String"

    value: bpy.props.StringProperty(name="")

    def init(self, context):
        self.outputs.new("NodeSocketString", "String")

    def draw_buttons(self, context, layout):
        layout.prop(self, "value")

    def execute(self, context):
        return {
            'String': self.value
        }

class NodeInteger(DreamTexturesNode):
    bl_idname = "dream_textures.node_integer"
    bl_label = "Integer"

    value: bpy.props.IntProperty(name="")

    def init(self, context):
        self.outputs.new("NodeSocketInt", "Integer")

    def draw_buttons(self, context, layout):
        layout.prop(self, "value")

    def execute(self, context):
        return {
            'Integer': self.value
        }

class NodeCollection(DreamTexturesNode):
    bl_idname = "dream_textures.node_collection"
    bl_label = "Collection"

    value: bpy.props.PointerProperty(type=bpy.types.Collection, name="")

    def init(self, context):
        self.outputs.new("NodeSocketCollection", "Collection")

    def draw_buttons(self, context, layout):
        layout.prop(self, "value")

    def execute(self, context):
        return {
            'Collection': self.value
        }

class NodeImage(DreamTexturesNode):
    bl_idname = "dream_textures.node_image"
    bl_label = "Image"

    value: bpy.props.PointerProperty(type=bpy.types.Image)

    def init(self, context):
        self.outputs.new("NodeSocketImage", "Image")

    def draw_buttons(self, context, layout):
        layout.prop(self, "value", text="")

    def execute(self, context):
        return {
            'Image': np.array(self.value.pixels).reshape((*self.value.size, self.value.channels))
        }

class NodeSceneInfo(DreamTexturesNode):
    bl_idname = "dream_textures.node_scene"
    bl_label = "Scene Info"

    def init(self, context):
        self.outputs.new("NodeSocketImage", "Depth Map")
        self.outputs.new("NodeSocketImage", "OpenPose Map")

    def draw_buttons(self, context, layout):
        pass

    @classmethod
    def render_depth_map(cls, context, collection=None):
        width, height = context.scene.render.resolution_x, context.scene.render.resolution_y
        matrix = context.scene.camera.matrix_world.inverted()
        projection_matrix = context.scene.camera.calc_matrix_camera(
            context,
            x=width,
            y=height
        )
        offscreen = gpu.types.GPUOffScreen(width, height)

        with offscreen.bind():
            fb = gpu.state.active_framebuffer_get()
            fb.clear(color=(0.0, 0.0, 0.0, 0.0))
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.depth_mask_set(True)
            with gpu.matrix.push_pop():
                gpu.matrix.load_matrix(matrix)
                gpu.matrix.load_projection_matrix(projection_matrix)
                
                shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')

                for object in (context.scene.objects if collection is None else collection.objects):
                    object = object.evaluated_get(context)
                    try:
                        mesh = object.to_mesh(depsgraph=context).copy()
                    except:
                        continue
                    if mesh is None:
                        continue
                    vertices = np.empty((len(mesh.vertices), 3), 'f')
                    indices = np.empty((len(mesh.loop_triangles), 3), 'i')

                    mesh.transform(object.matrix_world)
                    mesh.vertices.foreach_get("co", np.reshape(vertices, len(mesh.vertices) * 3))
                    mesh.loop_triangles.foreach_get("vertices", np.reshape(indices, len(mesh.loop_triangles) * 3))
                    
                    batch = batch_for_shader(
                        shader, 'TRIS',
                        {"pos": vertices},
                        indices=indices,
                    )
                    batch.draw(shader)
            depth = np.array(fb.read_depth(0, 0, width, height).to_list())
            depth = np.interp(depth, [np.ma.masked_equal(depth, 0, copy=False).min(), depth.max()], [0, 1]).clip(0, 1)
        offscreen.free()
        return depth

    def execute(self, context):
        return {
            'Depth Map': NodeSceneInfo.render_depth_map(context),
            'OpenPose Map': openpose.render_openpose_map(context)
        }