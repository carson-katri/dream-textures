import bpy
import bpy_extras
import gpu
from gpu_extras.batch import batch_for_shader
from gpu_extras.presets import draw_circle_2d
import mathutils
import math
import numpy as np
import enum
from ..node import DreamTexturesNode

def draw_circle_2d(center, radius, segments, color):
    m = (1.0 / (segments - 1)) * (math.pi * 2)

    coords = [
        (
            center[0] + math.cos(m * p) * radius,
            center[1] + math.sin(m * p) * radius
        )
        for p in range(segments)
    ]

    shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'TRI_FAN', {"pos": coords})
    shader.uniform_float("color", color)
    batch.draw(shader)

def draw_ellipse_2d(start, end, thickness, segments, color):
    length = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    theta = math.atan2(end[1] - start[1], end[0] - start[0])
    center = (
        (start[0] + end[0]) / 2,
        (start[1] + end[1]) / 2
    )
    major, minor = length / 2, thickness
    m = (1.0 / (segments - 1)) * (math.pi * 2)

    coords = [
        (
            center[0] + major * math.cos(m * p) * math.cos(theta) - minor * math.sin(m * p) * math.sin(theta),
            center[1] + major * math.cos(m * p) * math.sin(theta) + minor * math.sin(m * p) * math.cos(theta)
        )
        for p in range(segments)
    ]

    shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'TRI_FAN', {"pos": coords})
    shader.uniform_float("color", color)
    batch.draw(shader)

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
                    try:
                        mesh = object.to_mesh(depsgraph=context)
                    except:
                        continue
                    if mesh is None:
                        continue
                    vertices = np.empty((len(mesh.vertices), 3), 'f')
                    indices = np.empty((len(mesh.loop_triangles), 3), 'i')

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

    @classmethod
    def render_openpose_map(cls, context, collection=None):
        width, height = context.scene.render.resolution_x, context.scene.render.resolution_y
        offscreen = gpu.types.GPUOffScreen(width, height)

        with offscreen.bind():
            fb = gpu.state.active_framebuffer_get()
            fb.clear(color=(0.0, 0.0, 0.0, 0.0))
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.depth_mask_set(True)

            class Side(enum.IntEnum):
                HEAD = 0
                TAIL = 1
            
            class Bone(enum.IntEnum):
                NOSE = 0
                CHEST = 1

                SHOULDER_L = 2
                SHOULDER_R = 3
                ELBOW_L = 4
                ELBOW_R = 5
                HAND_L = 6
                HAND_R = 7

                HIP_L = 8
                HIP_R = 9
                KNEE_L = 10
                KNEE_R = 11
                FOOT_L = 12
                FOOT_R = 13

                EYE_L = 14
                EYE_R = 15

                EAR_L = 16
                EAR_R = 17

                def identify(self, pose):
                    options = self.name_detection_options()
                    for option in options:
                        if (result := pose.bones.get(option[0], None)) is not None:
                            return result, option[1]
                    return None, None

                def name_detection_options(self):
                    match self:
                        case Bone.NOSE:
                            return [('nose_ik.001', Side.TAIL), ('nose.001', Side.TAIL)]
                        case Bone.CHEST:
                            return [('spine_fk.003', Side.TAIL), ('spine.003', Side.TAIL)]
                        case Bone.SHOULDER_L:
                            return [('shoulder_ik.L', Side.TAIL), ('shoulder.L', Side.TAIL)]
                        case Bone.SHOULDER_R:
                            return [('shoulder_ik.R', Side.TAIL), ('shoulder.R', Side.TAIL)]
                        case Bone.ELBOW_L:
                            return [('upper_arm_ik.L', Side.TAIL), ('upper_arm.L', Side.TAIL)]
                        case Bone.ELBOW_R:
                            return [('upper_arm_ik.R', Side.TAIL), ('upper_arm.R', Side.TAIL)]
                        case Bone.HAND_L:
                            return [('hand_ik.L', Side.TAIL), ('forearm.L', Side.TAIL)]
                        case Bone.HAND_R:
                            return [('hand_ik.R', Side.TAIL), ('forearm.R', Side.TAIL)]
                        case Bone.HIP_L:
                            return [('thigh_ik.L', Side.HEAD), ('thigh.L', Side.HEAD)]
                        case Bone.HIP_R:
                            return [('thigh_ik.R', Side.HEAD), ('thigh.R', Side.HEAD)]
                        case Bone.KNEE_L:
                            return [('thigh_ik.L', Side.TAIL), ('thigh.L', Side.TAIL)]
                        case Bone.KNEE_R:
                            return [('thigh_ik.R', Side.TAIL), ('thigh.R', Side.TAIL)]
                        case Bone.FOOT_L:
                            return [('foot_ik.L', Side.TAIL), ('shin.L', Side.TAIL)]
                        case Bone.FOOT_R:
                            return [('foot_ik.R', Side.TAIL), ('shin.R', Side.TAIL)]
                        case Bone.EYE_L:
                            return [('master_eye.L', Side.TAIL), ('eye.L', Side.TAIL)]
                        case Bone.EYE_R:
                            return [('master_eye.R', Side.TAIL), ('eye.R', Side.TAIL)]
                        case Bone.EAR_L:
                            return [('ear.L', Side.TAIL), ('ear.L.001', Side.TAIL)]
                        case Bone.EAR_R:
                            return [('ear.R', Side.TAIL), ('ear.R.001', Side.TAIL)]
                        
                def color(self):
                    match self:
                        case Bone.NOSE: return (255, 0, 0)
                        case Bone.CHEST: return (255, 85, 0)
                        case Bone.SHOULDER_L: return (85, 255, 0)
                        case Bone.SHOULDER_R: return (255, 170, 0)
                        case Bone.ELBOW_L: return (0, 255, 0)
                        case Bone.ELBOW_R: return (255, 255, 0)
                        case Bone.HAND_L: return (0, 255, 85)
                        case Bone.HAND_R: return (170, 255, 0)
                        case Bone.HIP_L: return (0, 85, 255)
                        case Bone.HIP_R: return (0, 255, 170)
                        case Bone.KNEE_L: return (0, 0, 255)
                        case Bone.KNEE_R: return (0, 255, 255)
                        case Bone.FOOT_L: return (85, 0, 255)
                        case Bone.FOOT_R: return (0, 170, 255)
                        case Bone.EYE_L: return (255, 0, 255)
                        case Bone.EYE_R: return (170, 0, 255)
                        case Bone.EAR_L: return (255, 0, 85)
                        case Bone.EAR_R: return (255, 0, 170)

            lines = {
                (Bone.NOSE, Bone.CHEST): (0, 0, 255),
                (Bone.CHEST, Bone.SHOULDER_L): (255, 85, 0),
                (Bone.CHEST, Bone.SHOULDER_R): (255, 0, 0),
                (Bone.SHOULDER_L, Bone.ELBOW_L): (170, 255, 0),
                (Bone.SHOULDER_R, Bone.ELBOW_R): (255, 170, 0),
                (Bone.ELBOW_L, Bone.HAND_L): (85, 255, 0),
                (Bone.ELBOW_R, Bone.HAND_R): (255, 255, 0),
                (Bone.CHEST, Bone.HIP_L): (0, 255, 255),
                (Bone.CHEST, Bone.HIP_R): (0, 255, 0),
                (Bone.HIP_L, Bone.KNEE_L): (0, 170, 255),
                (Bone.HIP_R, Bone.KNEE_R): (0, 255, 85),
                (Bone.KNEE_L, Bone.FOOT_L): (0, 85, 255),
                (Bone.KNEE_R, Bone.FOOT_R): (0, 255, 170),
                (Bone.NOSE, Bone.EYE_L): (255, 0, 255),
                (Bone.NOSE, Bone.EYE_R): (85, 0, 255),
                (Bone.EYE_L, Bone.EAR_L): (255, 0, 170),
                (Bone.EYE_R, Bone.EAR_R): (170, 0, 255),
            }
                                
            with gpu.matrix.push_pop():
                gpu.matrix.load_matrix(mathutils.Matrix.Identity(4))
                gpu.matrix.load_projection_matrix(mathutils.Matrix.Identity(4))
                gpu.state.blend_set('ALPHA')

                shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
                batch = batch_for_shader(shader, 'TRI_STRIP', {"pos": [(-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0)]})
                shader.uniform_float("color", (0, 0, 0, 1))
                batch.draw(shader)

                for object in (context.scene.objects if collection is None else collection.objects):
                    if object.hide_render:
                        continue
                    if object.pose is None:
                        continue
                    for connection, color in lines.items():
                        a, a_side = connection[0].identify(object.pose)
                        b, b_side = connection[1].identify(object.pose)
                        if a is None or b is None:
                            continue
                        a = bpy_extras.object_utils.world_to_camera_view(context.scene, context.scene.camera, object.matrix_world @ (a.tail if a_side == Side.TAIL else a.head))
                        b = bpy_extras.object_utils.world_to_camera_view(context.scene, context.scene.camera, object.matrix_world @ (b.tail if b_side == Side.TAIL else b.head))
                        draw_ellipse_2d(((a[0] - 0.5) * 2, (a[1] - 0.5) * 2), ((b[0] - 0.5) * 2, (b[1] - 0.5) * 2), .015, 32, (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, 0.5))
                    for b in Bone:
                        bone, side = b.identify(object.pose)
                        color = b.color()
                        if bone is None: continue
                        tail = bpy_extras.object_utils.world_to_camera_view(context.scene, context.scene.camera, object.matrix_world @ (bone.tail if side == Side.TAIL else bone.head))
                        draw_circle_2d(((tail[0] - 0.5) * 2, (tail[1] - 0.5) * 2), .015, 16, (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, 0.5))

            depth = np.array(fb.read_color(0, 0, width, height, 4, 0, 'FLOAT').to_list())
        offscreen.free()
        return depth

    def execute(self, context):
        return {
            'Depth Map': NodeSceneInfo.render_depth_map(context),
            'OpenPose Map': NodeSceneInfo.render_openpose_map(context)
        }