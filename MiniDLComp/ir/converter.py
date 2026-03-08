from typing import Dict
import torch
import torch.fx as fx
from ir.ir_graph import IRGraph
from ir.ir_node import IRNode


SUPPORTED_MODULES = {
    torch.nn.Linear: "linear",
    torch.nn.ReLU: "relu",
}

SUPPORTED_FUNCTIONS = {
    torch.relu: "relu",
}


class FXToIRConverter:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.modules: Dict[str, torch.nn.Module] = dict(model.named_modules())

    def convert(self, example_input: torch.Tensor) -> IRGraph:
        traced = fx.symbolic_trace(self.model)
        graph = IRGraph()
        env: Dict[str, str] = {}

        for node in traced.graph.nodes:
            if node.op == "placeholder":
                ir_node = IRNode(name=node.name, op_type="input")
                graph.add_node(ir_node)
                graph.input_nodes.append(node.name)
                env[node.name] = node.name

            elif node.op == "get_attr":
                attr_value = self._resolve_attr(node.target)
                ir_node = IRNode(
                    name=node.name,
                    op_type="const",
                    value=attr_value,
                    attrs={"target": str(node.target)},
                )
                graph.add_node(ir_node)
                env[node.name] = node.name

            elif node.op == "call_module":
                module = self.modules[node.target]
                op_type = self._module_to_op(module)
                attrs = self._extract_module_attrs(module)
                ir_node = IRNode(name=node.name, op_type=op_type, attrs=attrs)
                graph.add_node(ir_node)
                for arg in node.args:
                    if hasattr(arg, "name") and arg.name in env:
                        graph.add_edge(env[arg.name], node.name)
                env[node.name] = node.name

            elif node.op == "call_function":
                op_type = SUPPORTED_FUNCTIONS.get(node.target, str(node.target))
                ir_node = IRNode(name=node.name, op_type=op_type)
                graph.add_node(ir_node)
                for arg in node.args:
                    if hasattr(arg, "name") and arg.name in env:
                        graph.add_edge(env[arg.name], node.name)
                env[node.name] = node.name

            elif node.op == "output":
                output_arg = node.args[0]
                if hasattr(output_arg, "name"):
                    graph.mark_output(env[output_arg.name])

        return graph

    def _module_to_op(self, module: torch.nn.Module) -> str:
        for module_type, op_name in SUPPORTED_MODULES.items():
            if isinstance(module, module_type):
                return op_name
        return type(module).__name__.lower()

    def _extract_module_attrs(self, module: torch.nn.Module) -> Dict[str, str]:
        attrs: Dict[str, str] = {}
        if isinstance(module, torch.nn.Linear):
            attrs["in_features"] = module.in_features
            attrs["out_features"] = module.out_features
            attrs["has_bias"] = module.bias is not None
        return attrs

    def _resolve_attr(self, target: str):
        attr_itr = self.model
        for atom in target.split("."):
            attr_itr = getattr(attr_itr, atom)
        return attr_itr
