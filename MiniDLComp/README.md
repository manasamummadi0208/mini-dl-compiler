# MiniDLComp

MiniDLComp is a lightweight deep learning compiler starter project that converts a simple PyTorch model into a custom intermediate representation (IR), applies compiler-style optimization passes, and executes the optimized graph.

## Features
- Converts a PyTorch FX graph into a custom IR
- Supports basic ops: input, const, linear, relu, linear_relu
- Implements optimization passes:
  - constant folding
  - dead code elimination
  - operator fusion
- Benchmarks original PyTorch execution against a simple IR executor

## How to Run
```bash
pip install -r requirements.txt
python main.py
```

## Future Improvements
- Add shape inference
- Add topological sorting
- Support more ops like add, matmul, conv2d
- Export graph as JSON
- Add ONNX import
- Add MLIR-inspired dialect structure
- Add cost model and scheduling
