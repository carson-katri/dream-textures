from .engine import *
from .node_tree import *
from .node_executor import *
from .node import *
from .nodes.input_nodes import *
from .nodes.pipeline_nodes import *
from .nodes.utility_nodes import *
from .nodes.annotation_nodes import *
from .annotations import openpose
from .annotations import ade20k

import bpy
import nodeitems_utils

class DreamTexturesNodeCategory(nodeitems_utils.NodeCategory):
    @classmethod
    def poll(cls, context):
        return context.space_data.tree_type == DreamTexturesNodeTree.__name__

pipeline_items = [
    nodeitems_utils.NodeItem(NodeStableDiffusion.bl_idname),
    nodeitems_utils.NodeItem(NodeControlNet.bl_idname),
]

input_items = [
    nodeitems_utils.NodeItem(NodeInteger.bl_idname),
    nodeitems_utils.NodeItem(NodeString.bl_idname),
    nodeitems_utils.NodeItem(NodeImage.bl_idname),
    nodeitems_utils.NodeItem(NodeCollection.bl_idname),
    nodeitems_utils.NodeItem(NodeRenderProperties.bl_idname),
]
if bpy.app.version >= (3, 5, 0):
    input_items.append(nodeitems_utils.NodeItem(NodeImageFile.bl_idname))

utility_items = [
    nodeitems_utils.NodeItem(NodeMath.bl_idname),
    nodeitems_utils.NodeItem(NodeRandomValue.bl_idname),
    nodeitems_utils.NodeItem(NodeRandomSeed.bl_idname),
    nodeitems_utils.NodeItem(NodeSeed.bl_idname),
    nodeitems_utils.NodeItem(NodeClamp.bl_idname),
    nodeitems_utils.NodeItem(NodeFramePath.bl_idname),
    nodeitems_utils.NodeItem(NodeCropImage.bl_idname),
    nodeitems_utils.NodeItem(NodeJoinImages.bl_idname),
    nodeitems_utils.NodeItem(NodeColorCorrect.bl_idname),
    nodeitems_utils.NodeItem(NodeSeparateColor.bl_idname),
    nodeitems_utils.NodeItem(NodeCombineColor.bl_idname),
    nodeitems_utils.NodeItem(NodeSwitch.bl_idname),
    nodeitems_utils.NodeItem(NodeCompare.bl_idname),
    nodeitems_utils.NodeItem(NodeReplaceString.bl_idname),
]
if bpy.app.version >= (3, 5, 0):
    utility_items.append(nodeitems_utils.NodeItem(NodeResizeImage.bl_idname))

annotations_items = [
    nodeitems_utils.NodeItem(NodeAnnotationDepth.bl_idname),
    nodeitems_utils.NodeItem(NodeAnnotationOpenPose.bl_idname),
    nodeitems_utils.NodeItem(NodeAnnotationADE20K.bl_idname),
    nodeitems_utils.NodeItem(NodeAnnotationViewport.bl_idname),
]

group_items = [
    nodeitems_utils.NodeItem(bpy.types.NodeGroupOutput.__name__),
]

categories = [
    DreamTexturesNodeCategory("DREAM_TEXTURES_PIPELINE", "Pipeline", items=pipeline_items),
    DreamTexturesNodeCategory("DREAM_TEXTURES_INPUT", "Input", items=input_items),
    DreamTexturesNodeCategory("DREAM_TEXTURES_UTILITY", "Utilities", items=utility_items),
    DreamTexturesNodeCategory("DREAM_TEXTURES_ANNOTATIONS", "Annotations", items=annotations_items),
    DreamTexturesNodeCategory("DREAM_TEXTURES_GROUP", "Group", items=group_items),
]

def register():
    # Prompt
    bpy.types.Scene.dream_textures_engine_prompt = bpy.props.PointerProperty(type=DreamPrompt)
    
    # OpenPose
    bpy.utils.register_class(openpose.ArmatureOpenPoseData)
    bpy.types.Armature.dream_textures_openpose = bpy.props.PointerProperty(
        type=openpose.ArmatureOpenPoseData
    )
    bpy.utils.register_class(openpose.BoneOpenPoseData)
    bpy.types.Bone.dream_textures_openpose = bpy.props.PointerProperty(
        type=openpose.BoneOpenPoseData
    )

    # ADE20K
    bpy.utils.register_class(ade20k.ObjectADE20KData)
    bpy.types.Object.dream_textures_ade20k = bpy.props.PointerProperty(
        type=ade20k.ObjectADE20KData
    )

    bpy.utils.register_class(DreamTexturesNodeTree)
    
    # Nodes
    bpy.utils.register_class(NodeSocketControlNet)
    bpy.utils.register_class(NodeStableDiffusion)
    bpy.utils.register_class(NodeControlNet)

    bpy.utils.register_class(NodeInteger)
    bpy.utils.register_class(NodeString)
    bpy.utils.register_class(NodeCollection)
    bpy.utils.register_class(NodeImage)
    bpy.utils.register_class(NodeImageFile)
    bpy.utils.register_class(NodeRenderProperties)
    
    bpy.utils.register_class(NodeAnnotationDepth)
    bpy.utils.register_class(NodeAnnotationNormal)
    bpy.utils.register_class(NodeAnnotationOpenPose)
    bpy.utils.register_class(NodeAnnotationADE20K)
    bpy.utils.register_class(NodeAnnotationViewport)

    bpy.utils.register_class(NodeMath)
    bpy.utils.register_class(NodeRandomValue)
    bpy.utils.register_class(NodeRandomSeed)
    bpy.utils.register_class(NodeSeed)
    bpy.utils.register_class(NodeClamp)
    bpy.utils.register_class(NodeFramePath)
    bpy.utils.register_class(NodeCropImage)
    bpy.utils.register_class(NodeResizeImage)
    bpy.utils.register_class(NodeJoinImages)
    bpy.utils.register_class(NodeColorCorrect)
    bpy.utils.register_class(NodeSeparateColor)
    bpy.utils.register_class(NodeCombineColor)
    bpy.utils.register_class(NodeSwitch)
    bpy.utils.register_class(NodeCompare)
    bpy.utils.register_class(NodeReplaceString)

    nodeitems_utils.register_node_categories("DREAM_TEXTURES_CATEGORIES", categories)

def unregister():
    # OpenPose
    del bpy.types.Armature.dream_textures_openpose
    bpy.utils.unregister_class(openpose.ArmatureOpenPoseData)
    del bpy.types.Bone.dream_textures_openpose
    bpy.utils.unregister_class(openpose.BoneOpenPoseData)

    # ADE20K
    del bpy.types.Object.dream_textures_ade20k
    bpy.utils.unregister_class(ade20k.ObjectADE20KData)

    bpy.utils.unregister_class(DreamTexturesNodeTree)
    
    # Nodes
    bpy.utils.unregister_class(NodeSocketControlNet)
    bpy.utils.unregister_class(NodeStableDiffusion)
    bpy.utils.unregister_class(NodeControlNet)

    bpy.utils.unregister_class(NodeInteger)
    bpy.utils.unregister_class(NodeString)
    bpy.utils.unregister_class(NodeCollection)
    bpy.utils.unregister_class(NodeImage)
    bpy.utils.unregister_class(NodeImageFile)
    bpy.utils.unregister_class(NodeRenderProperties)

    bpy.utils.unregister_class(NodeAnnotationDepth)
    bpy.utils.unregister_class(NodeAnnotationNormal)
    bpy.utils.unregister_class(NodeAnnotationOpenPose)
    bpy.utils.unregister_class(NodeAnnotationADE20K)
    bpy.utils.unregister_class(NodeAnnotationViewport)
    
    bpy.utils.unregister_class(NodeMath)
    bpy.utils.unregister_class(NodeRandomValue)
    bpy.utils.unregister_class(NodeRandomSeed)
    bpy.utils.unregister_class(NodeSeed)
    bpy.utils.unregister_class(NodeClamp)
    bpy.utils.unregister_class(NodeFramePath)
    bpy.utils.unregister_class(NodeCropImage)
    bpy.utils.unregister_class(NodeResizeImage)
    bpy.utils.unregister_class(NodeJoinImages)
    bpy.utils.unregister_class(NodeColorCorrect)
    bpy.utils.unregister_class(NodeSeparateColor)
    bpy.utils.unregister_class(NodeCombineColor)
    bpy.utils.unregister_class(NodeSwitch)
    bpy.utils.unregister_class(NodeCompare)
    bpy.utils.unregister_class(NodeReplaceString)

    nodeitems_utils.unregister_node_categories("DREAM_TEXTURES_CATEGORIES")