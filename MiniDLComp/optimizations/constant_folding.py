from ir.ir_graph import IRGraph


class ConstantFoldingPass:
    def run(self, graph: IRGraph) -> bool:
        changed = False
        # Minimal starter version:
        # If a relu consumes a constant tensor, precompute it.
        for node in graph.live_nodes():
            if node.op_type == "relu" and len(node.inputs) == 1:
                input_node = graph.get_node(node.inputs[0])
                if input_node.is_constant() and input_node.value is not None:
                    node.op_type = "const"
                    node.value = input_node.value.clamp(min=0)
                    node.inputs = []
                    changed = True
        return changed
