import bpy
# from dream_textures.engine import node_executor
# node_executor.execute(bpy.data.node_groups["NodeTree"], bpy.context.evaluated_depsgraph_get())

class NodeExecutionContext:
    def __init__(self, depsgraph, start, update, end, test_break, cache={}):
        self.depsgraph = depsgraph
        self.start = start
        self.update = update
        self.end = end
        self.test_break = test_break
        self.cache = {}
        self.preferences = bpy.context.preferences
    
    def _evaluate_input(self, input):
        if input.is_linked:
            if len(input.links) > 1:
                return [
                    self.execute(link.from_socket.node)[link.from_socket.name]
                    for link in input.links
                ]
            else:
                return self.execute(input.links[0].from_socket.node)[input.links[0].from_socket.name]
        else:
            return getattr(input, 'default_value', None)

    def execute(self, node):
        if self.test_break():
            return None
        result = None
        match node.bl_idname:
            case 'dream_textures.node_switch':
                kwargs = {
                    'switch': self._evaluate_input(node.inputs[0]),
                    'false': lambda: self._evaluate_input(node.inputs[1]),
                    'true': lambda: self._evaluate_input(node.inputs[2])
                }
                self.start(node)
                result = node.execute(self, **kwargs)
            case _:
                match node.type:
                    case 'GROUP_INPUT':
                        self.start(node)
                        result = {
                            input.name: input.default_value
                            for input in self.depsgraph.scene.dream_textures_render_engine.node_tree.inputs
                        }
                    case 'GROUP_OUTPUT':
                        self.start(node)
                        result = [
                            (input.name, self.execute(input.links[0].from_socket.node)[input.links[0].from_socket.name])
                            for input in node.inputs if len(input.links) > 0
                        ]
                    case 'FRAME':
                        self.start(node)
                        result = None
                    case _:
                        if node in self.cache:
                            self.start(node)
                            result = self.cache[node]
                        else:
                            kwargs = {
                                input.name.lower().replace(' ', '_'): self._evaluate_input(input)
                                for input in node.inputs
                            }
                            self.start(node)
                            result = node.execute(self, **kwargs)
        self.end(node)
        return result

def execute(node_tree, depsgraph, node_begin=lambda node: None, node_update=lambda result: None, node_end=lambda node: None, test_break=lambda: False):
    output = next(n for n in node_tree.nodes if n.type == 'GROUP_OUTPUT')
    context = NodeExecutionContext(depsgraph, node_begin, node_update, node_end, test_break)
    return context.execute(output)