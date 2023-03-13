import bpy
import numpy as np
# from dream_textures.engine import node_executor
# node_executor.execute(bpy.data.node_groups["NodeTree"], bpy.context)

def execute_node(node, context, cache, on_execute=lambda _, __: None):
    if node in cache:
        return cache[node]
    kwargs = {
        input.name.lower().replace(' ', '_'): ([
            execute_node(link.from_socket.node, context, cache)[link.from_socket.name]
            for link in input.links
        ] if len(input.links) > 1 else execute_node(input.links[0].from_socket.node, context, cache)[input.links[0].from_socket.name])
        if input.is_linked else getattr(input, 'default_value', None)
        for input in node.inputs
    }
    if node.type == 'GROUP_OUTPUT':
        return list(kwargs.values())[0]
    result = node.execute(context, **kwargs)
    on_execute(node, result)
    cache[node] = result
    return result

def execute(node_tree, context, on_execute=lambda _, __: None):
    output = next(n for n in node_tree.nodes if n.type == 'GROUP_OUTPUT')
    cache = {}
    result = execute_node(output, context, cache, on_execute)
    return result