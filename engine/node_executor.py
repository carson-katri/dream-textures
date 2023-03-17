import bpy
import numpy as np
from threading import Event
import graphlib
# from dream_textures.engine import node_executor
# node_executor.execute(bpy.data.node_groups["NodeTree"], bpy.context.evaluated_depsgraph_get())

class NodeExecutionContext:
    def __init__(self, depsgraph, update):
        self.depsgraph = depsgraph
        self.update = update

def execute_node(node, context, cache):
    result = None
    match node.type:
        case 'GROUP_INPUT':
            return {
                input.name: input.default_value
                for input in context.depsgraph.scene.dream_textures_render_engine.node_tree.inputs
            }
        case 'GROUP_OUTPUT':
            return cache[node.inputs[0].links[0].from_socket.node]
        case _:
            if node in cache:
                return cache[node]
            kwargs = {
                input.name.lower().replace(' ', '_'): ([
                    cache[link.from_socket.node][link.from_socket.name]
                    for link in input.links
                ] if len(input.links) > 1 else cache[input.links[0].from_socket.node][input.links[0].from_socket.name])
                if input.is_linked else getattr(input, 'default_value', None)
                for input in node.inputs
            }
            if node.type == 'GROUP_OUTPUT':
                return list(kwargs.values())[0]
            result = node.execute(context, **kwargs)
            return result

def execute(node_tree, depsgraph, node_begin=lambda node: None, node_update=lambda result: None, node_end=lambda node: None):
    output = next(n for n in node_tree.nodes if n.type == 'GROUP_OUTPUT')
    cache = {}
    graph = {
        node: [link.from_socket.node for input in node.inputs for link in input.links]
        for node in node_tree.nodes 
    }
    sort = graphlib.TopologicalSorter(graph)
    for node in sort.static_order():
        if len(node.outputs) > 0 and next((l for i in node.outputs for l in i.links), None) is None:
            continue # node outputs are unused
        node_begin(node)
        result = execute_node(node, NodeExecutionContext(depsgraph, node_update), cache)
        cache[node] = result
        node_end(node)
    return next(iter(cache[output].values()))