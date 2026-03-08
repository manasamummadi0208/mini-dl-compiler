from typing import Dict, List, Optional
from ir.ir_node import IRNode


class IRGraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, IRNode] = {}
        self.input_nodes: List[str] = []
        self.output_node: Optional[str] = None

    def add_node(self, node: IRNode) -> None:
        if node.name in self.nodes:
            raise ValueError(f"Duplicate node name: {node.name}")
        self.nodes[node.name] = node

    def get_node(self, name: str) -> IRNode:
        return self.nodes[name]

    def add_edge(self, src: str, dst: str) -> None:
        if src not in self.nodes or dst not in self.nodes:
            raise KeyError("Both src and dst must exist before adding an edge")
        self.nodes[dst].inputs.append(src)
        self.nodes[src].users.append(dst)

    def mark_output(self, name: str) -> None:
        if name not in self.nodes:
            raise KeyError(f"Output node {name} not found")
        self.output_node = name

    def live_nodes(self) -> List[IRNode]:
        return [node for node in self.nodes.values() if not node.deleted]

    def summary(self) -> str:
        lines = ["IRGraph Summary:"]
        for node in self.live_nodes():
            lines.append(
                f"- {node.name}: {node.op_type}, inputs={node.inputs}, users={node.users}, attrs={node.attrs}"
            )
        lines.append(f"Output: {self.output_node}")
        return "\n".join(lines)
