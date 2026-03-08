import copy
import torch

from models.simple_mlp import SimpleMLP
from ir.converter import FXToIRConverter
from optimizations.pass_manager import PassManager
from runtime.executor import IRExecutor
from runtime.benchmark import benchmark_model, benchmark_executor


def main() -> None:
    torch.manual_seed(42)

    model = SimpleMLP()
    model.eval()
    x = torch.randn(1, 16)

    print("=== Building IR from PyTorch model ===")
    converter = FXToIRConverter(model)
    ir_graph = converter.convert(x)
    print(ir_graph.summary())

    original_graph = copy.deepcopy(ir_graph)

    print("\n=== Running optimization passes ===")
    pass_manager = PassManager()
    pass_manager.run(ir_graph)
    print(ir_graph.summary())

    print("\n=== Running outputs ===")
    with torch.no_grad():
        model_out = model(x)

    executor = IRExecutor(model)
    ir_out = executor.run(ir_graph, x)

    print("Model output:")
    print(model_out)
    print("\nIR output:")
    print(ir_out)

    max_diff = (model_out - ir_out).abs().max().item()
    print(f"\nMax absolute difference: {max_diff:.8f}")

    print("\n=== Benchmarking ===")
    model_time = benchmark_model(model, x)
    ir_time = benchmark_executor(executor, ir_graph, x)

    print(f"PyTorch model avg latency: {model_time * 1e6:.2f} us")
    print(f"IR executor avg latency:   {ir_time * 1e6:.2f} us")

    print("\n=== Notes ===")
    print("This starter project is meant to demonstrate compiler concepts:")
    print("- custom IR")
    print("- graph optimization passes")
    print("- fused operators")
    print("- simple execution backend")


if __name__ == "__main__":
    main()
