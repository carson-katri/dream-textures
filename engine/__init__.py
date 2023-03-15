from .engine import *
from .node_tree import *
from .node_executor import *
from .node import *
from .nodes.input_nodes import *
from .nodes.pipeline_nodes import *
from .nodes.utility_nodes import *
from .annotations import openpose

import bpy
import nodeitems_utils

class DreamTexturesNodeCategory(nodeitems_utils.NodeCategory):
    @classmethod
    def poll(cls, context):
        return context.space_data.tree_type == DreamTexturesNodeTree.__name__

categories = [
    DreamTexturesNodeCategory("DREAM_TEXTURES_PIPELINE", "Pipeline", items = [
        nodeitems_utils.NodeItem(NodeStableDiffusion.bl_idname),
    ]),
    DreamTexturesNodeCategory("DREAM_TEXTURES_INPUT", "Input", items = [
        nodeitems_utils.NodeItem(NodeInteger.bl_idname),
        nodeitems_utils.NodeItem(NodeString.bl_idname),
        nodeitems_utils.NodeItem(NodeImage.bl_idname),
        nodeitems_utils.NodeItem(NodeCollection.bl_idname),
        nodeitems_utils.NodeItem(NodeSceneInfo.bl_idname),
    ]),
    DreamTexturesNodeCategory("DREAM_TEXTURES_UTILITY", "Utilities", items = [
        nodeitems_utils.NodeItem(NodeMath.bl_idname),
        nodeitems_utils.NodeItem(NodeRandomValue.bl_idname),
    ]),
    DreamTexturesNodeCategory("DREAM_TEXTURES_GROUP", "Group", items = [
        nodeitems_utils.NodeItem(bpy.types.NodeGroupOutput.__name__),
    ]),
]

def register():
    # Prompt
    bpy.types.Scene.dream_textures_engine_prompt = bpy.props.PointerProperty(type=DreamPrompt)
    
    # Bone
    bpy.types.Bone.dream_textures_openpose = bpy.props.BoolProperty(
        name="Use OpenPose",
        default=False
    )
    bpy.types.Bone.dream_textures_openpose_bone = bpy.props.EnumProperty(
        name="OpenPose Bone",
        items=((str(b.value), b.name.title(), '') for b in openpose.Bone)
    )
    bpy.types.Bone.dream_textures_openpose_bone_side = bpy.props.EnumProperty(
        name="Endpoint Side",
        items=((str(s.value), s.name.title(), '') for s in openpose.Side)
    )

    bpy.utils.register_class(DreamTexturesNodeTree)
    
    # Nodes
    bpy.utils.register_class(NodeSocketControlNet)
    bpy.utils.register_class(NodeStableDiffusion)
    bpy.utils.register_class(NodeControlNet)

    bpy.utils.register_class(NodeInteger)
    bpy.utils.register_class(NodeString)
    bpy.utils.register_class(NodeCollection)
    bpy.utils.register_class(NodeSceneInfo)
    bpy.utils.register_class(NodeImage)

    bpy.utils.register_class(NodeMath)
    bpy.utils.register_class(NodeRandomValue)

    nodeitems_utils.register_node_categories("DREAM_TEXTURES_CATEGORIES", categories)

def unregister():
    bpy.utils.unregister_class(DreamTexturesNodeTree)
    
    # Nodes
    bpy.utils.unregister_class(NodeSocketControlNet)
    bpy.utils.unregister_class(NodeStableDiffusion)
    bpy.utils.unregister_class(NodeControlNet)

    bpy.utils.unregister_class(NodeInteger)
    bpy.utils.unregister_class(NodeString)
    bpy.utils.unregister_class(NodeCollection)
    bpy.utils.unregister_class(NodeSceneInfo)
    bpy.utils.unregister_class(NodeImage)
    
    bpy.utils.unregister_class(NodeMath)
    bpy.utils.unregister_class(NodeRandomValue)
    

    nodeitems_utils.unregister_node_categories("DREAM_TEXTURES_CATEGORIES")