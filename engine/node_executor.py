import graphlib
# from dream_textures.engine import node_executor
# node_executor.execute(bpy.data.node_groups["NodeTree"], bpy.context.evaluated_depsgraph_get())

class NodeExecutionContext:
    def __init__(self, depsgraph, update, test_break):
        self.depsgraph = depsgraph
        self.update = update
        self.test_break = test_break

def execute_node(node, context, cache):
    result = None
    match node.type:
        case 'GROUP_INPUT':
            return {
                input.name: input.default_value
                for input in context.depsgraph.scene.dream_textures_render_engine.node_tree.inputs
            }
        case 'GROUP_OUTPUT':
            return [
                (input.name, cache[input.links[0].from_socket.node][input.links[0].from_socket.name])
                for input in node.inputs if len(input.links) > 0
            ]
        case 'FRAME':
            return None
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

def execute(node_tree, depsgraph, node_begin=lambda node: None, node_update=lambda result: None, node_end=lambda node: None, test_break=lambda: False):
    output = next(n for n in node_tree.nodes if n.type == 'GROUP_OUTPUT')
    cache = {}
    graph = {
        node: [link.from_socket.node for input in node.inputs for link in input.links]
        for node in node_tree.nodes 
    }
    sort = graphlib.TopologicalSorter(graph)
    def is_used(node):
        if node.type == 'GROUP_OUTPUT':
            return True
        for output in node.outputs:
            for link in output.links:
                # if link.to_socket.node.bl_idname == 'dream_textures.node_switch':
                #     input = link.to_socket.node.inputs[0]
                #     switch_result = cache[input.links[0].from_socket.node][input.links[0].from_socket.name] if input.is_linked else input.default_value
                #     return (switch_result and link.to_socket.name == 'True') or (not switch_result and link.to_socket.name == 'False')
                if is_used(link.to_socket.node):
                    return True
        return False
    for node in sort.static_order():
        if test_break():
            return None
        if not is_used(node):
            continue # node outputs are unused
        node_begin(node)
        result = execute_node(node, NodeExecutionContext(depsgraph, node_update, test_break), cache)
        cache[node] = result
        node_end(node)
    return cache[output]