import time
import torch


def benchmark_model(model: torch.nn.Module, x: torch.Tensor, runs: int = 200) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            _ = model(x)
        start = time.perf_counter()
        for _ in range(runs):
            _ = model(x)
        end = time.perf_counter()
    return (end - start) / runs


def benchmark_executor(executor, graph, x: torch.Tensor, runs: int = 200) -> float:
    with torch.no_grad():
        for _ in range(20):
            _ = executor.run(graph, x)
        start = time.perf_counter()
        for _ in range(runs):
            _ = executor.run(graph, x)
        end = time.perf_counter()
    return (end - start) / runs
