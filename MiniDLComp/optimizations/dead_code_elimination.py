from typing import Set
from ir.ir_graph import IRGraph


class DeadCodeEliminationPass:
    def run(self, graph: IRGraph) -> bool:
        if graph.output_node is None:
            return False

        reachable: Set[str] = set()

        def dfs(node_name: str) -> None:
            if node_name in reachable:
                return
            reachable.add(node_name)
            node = graph.get_node(node_name)
            for inp in node.inputs:
                dfs(inp)

        dfs(graph.output_node)
        changed = False
        for node in graph.live_nodes():
            if node.name not in reachable and node.op_type != "input":
                node.deleted = True
                changed = True
        return changed
