from typing import Dict
import torch
from ir.ir_graph import IRGraph


class IRExecutor:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.module_map = dict(model.named_modules())

    def run(self, graph: IRGraph, input_tensor: torch.Tensor) -> torch.Tensor:
        values: Dict[str, torch.Tensor] = {}

        for input_name in graph.input_nodes:
            values[input_name] = input_tensor

        for node in graph.live_nodes():
            if node.op_type == "input":
                continue
            if node.op_type == "const":
                values[node.name] = node.value
            elif node.op_type == "linear":
                x = values[node.inputs[0]]
                module = self.module_map[node.name]
                values[node.name] = module(x)
            elif node.op_type == "relu":
                x = values[node.inputs[0]]
                values[node.name] = torch.relu(x)
            elif node.op_type == "linear_relu":
                x = values[node.inputs[0]]
                module = self.module_map[node.name]
                values[node.name] = torch.relu(module(x))
            else:
                raise NotImplementedError(f"Unsupported op_type: {node.op_type}")

        if graph.output_node is None:
            raise ValueError("Graph has no output node")
        return values[graph.output_node]
