from ir.ir_graph import IRGraph
from optimizations.constant_folding import ConstantFoldingPass
from optimizations.dead_code_elimination import DeadCodeEliminationPass
from optimizations.operator_fusion import OperatorFusionPass


class PassManager:
    def __init__(self) -> None:
        self.passes = [
            ConstantFoldingPass(),
            OperatorFusionPass(),
            DeadCodeEliminationPass(),
        ]

    def run(self, graph: IRGraph) -> None:
        iteration = 0
        while True:
            iteration += 1
            changed = False
            for optimization_pass in self.passes:
                changed = optimization_pass.run(graph) or changed
            if not changed or iteration > 5:
                break
