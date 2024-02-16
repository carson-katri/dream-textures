import bpy
import bpy_extras
import gpu
from gpu_extras.batch import batch_for_shader
import mathutils
import numpy as np
import enum
import math
import threading
from .compat import UNIFORM_COLOR

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

    def identify(self, armature, pose):
        if not getattr(armature.dream_textures_openpose, self.name):
            return None, None
        for bone in pose.bones:
            if bone.bone.dream_textures_openpose.enabled:
                if bone.bone.dream_textures_openpose.bone == str(self.value):
                    return bone, Side(int(bone.bone.dream_textures_openpose.side))
        options = self.name_detection_options()
        for option in options:
            if (result := pose.bones.get(option[0], None)) is not None:
                return result, option[1]
        return None, None

    def name_detection_options(self):
        match self:
            case Bone.NOSE:
                return [('nose_master', Side.TAIL), ('nose.001', Side.TAIL), ('Head', Side.TAIL)]
            case Bone.CHEST:
                return [('spine_fk.003', Side.TAIL), ('spine.003', Side.TAIL), ('Spine4', Side.TAIL)]
            case Bone.SHOULDER_L:
                return [('shoulder_ik.L', Side.TAIL), ('shoulder.L', Side.TAIL), ('LeftShoulder', Side.TAIL)]
            case Bone.SHOULDER_R:
                return [('shoulder_ik.R', Side.TAIL), ('shoulder.R', Side.TAIL), ('RightShoulder', Side.TAIL)]
            case Bone.ELBOW_L:
                return [('upper_arm_ik.L', Side.TAIL), ('upper_arm.L', Side.TAIL), ('LeftArm', Side.TAIL)]
            case Bone.ELBOW_R:
                return [('upper_arm_ik.R', Side.TAIL), ('upper_arm.R', Side.TAIL), ('RightArm', Side.TAIL)]
            case Bone.HAND_L:
                return [('hand_ik.L', Side.TAIL), ('forearm.L', Side.TAIL), ('LeftForeArm', Side.TAIL)]
            case Bone.HAND_R:
                return [('hand_ik.R', Side.TAIL), ('forearm.R', Side.TAIL), ('RightForeArm', Side.TAIL)]
            case Bone.HIP_L:
                return [('thigh_ik.L', Side.HEAD), ('thigh.L', Side.HEAD), ('LeftThigh', Side.HEAD)]
            case Bone.HIP_R:
                return [('thigh_ik.R', Side.HEAD), ('thigh.R', Side.HEAD), ('RightThigh', Side.HEAD)]
            case Bone.KNEE_L:
                return [('thigh_ik.L', Side.TAIL), ('thigh.L', Side.TAIL), ('LeftShin', Side.HEAD)]
            case Bone.KNEE_R:
                return [('thigh_ik.R', Side.TAIL), ('thigh.R', Side.TAIL), ('RightShin', Side.HEAD)]
            case Bone.FOOT_L:
                return [('foot_ik.L', Side.TAIL), ('shin.L', Side.TAIL), ('LeftFoot', Side.HEAD)]
            case Bone.FOOT_R:
                return [('foot_ik.R', Side.TAIL), ('shin.R', Side.TAIL), ('RightFoot', Side.HEAD)]
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

openpose_bones = ((str(b.value), b.name.title(), '') for b in Bone)
openpose_sides = ((str(s.value), s.name.title(), '') for s in Side)
class BoneOpenPoseData(bpy.types.PropertyGroup):
    bl_label = "OpenPose"
    bl_idname = "dream_textures.BoneOpenPoseData"

    enabled: bpy.props.BoolProperty(name="Enabled", default=False)
    bone: bpy.props.EnumProperty(
        name="OpenPose Bone",
        items=openpose_bones
    )
    side: bpy.props.EnumProperty(
        name="Bone Side",
        items=openpose_sides
    )

ArmatureOpenPoseData = type('ArmatureOpenPoseData', (bpy.types.PropertyGroup,), {
    "bl_label": "OpenPose",
    "bl_idname": "dream_textures.ArmatureOpenPoseData",
    "__annotations__": { b.name: bpy.props.BoolProperty(name=b.name.title(), default=True) for b in Bone },
})

def render_openpose_map(context, collection=None):
    e = threading.Event()
    result = None
    def _execute():
        nonlocal result
        width, height = context.scene.render.resolution_x, context.scene.render.resolution_y
        offscreen = gpu.types.GPUOffScreen(width, height)

        with offscreen.bind():
            fb = gpu.state.active_framebuffer_get()
            fb.clear(color=(0.0, 0.0, 0.0, 0.0), depth=1)
            gpu.state.depth_test_set('LESS_EQUAL')
            gpu.state.depth_mask_set(True)

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
                ratio = width / height
                projection_matrix = mathutils.Matrix((
                    (1 / ratio, 0, 0, 0),
                    (0, 1, 0, 0),
                    (0, 0, -1, 0),
                    (0, 0, 0, 1)
                ))
                gpu.matrix.load_matrix(mathutils.Matrix.Identity(4))
                gpu.matrix.load_projection_matrix(projection_matrix)
                gpu.state.blend_set('ALPHA')

                def transform(x, y):
                    return (
                        (x - 0.5) * 2 * ratio,
                        (y - 0.5) * 2
                    )

                shader = gpu.shader.from_builtin(UNIFORM_COLOR)
                batch = batch_for_shader(shader, 'TRI_STRIP', {"pos": [(-ratio, -1, 0), (-ratio, 1, 0), (ratio, -1, 0), (ratio, 1, 0)]})
                shader.bind()
                shader.uniform_float("color", (0, 0, 0, 1))
                batch.draw(shader)

                for object in (context.scene.objects if collection is None else collection.objects):
                    object = object.evaluated_get(context)
                    if object.hide_render:
                        continue
                    if object.pose is None:
                        continue
                    for connection, color in lines.items():
                        a, a_side = connection[0].identify(object.data, object.pose)
                        b, b_side = connection[1].identify(object.data, object.pose)
                        if a is None or b is None:
                            continue
                        a = bpy_extras.object_utils.world_to_camera_view(context.scene, context.scene.camera, object.matrix_world @ (a.tail if a_side == Side.TAIL else a.head))
                        b = bpy_extras.object_utils.world_to_camera_view(context.scene, context.scene.camera, object.matrix_world @ (b.tail if b_side == Side.TAIL else b.head))
                        draw_ellipse_2d(transform(a[0], a[1]), transform(b[0], b[1]), .015, 32, (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, 0.5))
                    for b in Bone:
                        bone, side = b.identify(object.data, object.pose)
                        color = b.color()
                        if bone is None: continue
                        tail = bpy_extras.object_utils.world_to_camera_view(context.scene, context.scene.camera, object.matrix_world @ (bone.tail if side == Side.TAIL else bone.head))
                        draw_circle_2d(transform(tail[0], tail[1]), .015, 16, (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, 0.5))

            depth = np.array(fb.read_color(0, 0, width, height, 4, 0, 'FLOAT').to_list())
        gpu.state.depth_test_set('NONE')
        offscreen.free()
        result = depth
        e.set()
    if threading.current_thread() == threading.main_thread():
        _execute()
        return result
    else:
        bpy.app.timers.register(_execute, first_interval=0)
        e.wait()
        return result

def draw_circle_2d(center, radius, segments, color):
    m = (1.0 / (segments - 1)) * (math.pi * 2)

    coords = [
        (
            center[0] + math.cos(m * p) * radius,
            center[1] + math.sin(m * p) * radius,
            0
        )
        for p in range(segments)
    ]

    shader = gpu.shader.from_builtin(UNIFORM_COLOR)
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
            center[1] + major * math.cos(m * p) * math.sin(theta) + minor * math.sin(m * p) * math.cos(theta),
            0
        )
        for p in range(segments)
    ]

    shader = gpu.shader.from_builtin(UNIFORM_COLOR)
    batch = batch_for_shader(shader, 'TRI_FAN', {"pos": coords})
    shader.uniform_float("color", color)
    batch.draw(shader)