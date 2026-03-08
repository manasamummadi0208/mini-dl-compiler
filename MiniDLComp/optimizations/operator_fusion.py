from ir.ir_graph import IRGraph


class OperatorFusionPass:
    def run(self, graph: IRGraph) -> bool:
        changed = False

        for node in graph.live_nodes():
            if node.deleted or node.op_type != "relu" or len(node.inputs) != 1:
                continue

            parent = graph.get_node(node.inputs[0])
            if parent.deleted:
                continue

            if parent.op_type == "linear" and len(parent.users) == 1 and parent.users[0] == node.name:
                parent.op_type = "linear_relu"
                node.deleted = True

                # redirect users of relu to fused parent
                for user_name in list(node.users):
                    user = graph.get_node(user_name)
                    user.inputs = [parent.name if x == node.name else x for x in user.inputs]
                    parent.users.append(user_name)

                parent.users = [u for u in parent.users if u != node.name]

                if graph.output_node == node.name:
                    graph.output_node = parent.name

                changed = True

        return changed
