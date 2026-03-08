# MiniDLComp – Lightweight Deep Learning Compiler

MiniDLComp is a lightweight deep learning compiler that converts a PyTorch model into a custom intermediate representation (IR), applies compiler-style optimization passes, and executes the optimized graph.

The goal of this project is to demonstrate how machine learning models can be compiled and optimized before execution, similar to how ML compilers like TVM, Glow, and MLIR operate.

---

## System Architecture

PyTorch Model
      ↓
FX Graph Extraction
      ↓
Custom Intermediate Representation (IR)
      ↓
Optimization Passes
  • Constant Folding
  • Dead Code Elimination
  • Operator Fusion
      ↓
Optimized IR Graph
      ↓
Custom Execution Engine
      ↓
Optimized Model Inference

---

## Features

• Converts a PyTorch FX computation graph into a custom IR  
• Supports core operations:
  - input
  - const
  - linear
  - relu
  - linear_relu  

• Implements compiler-style optimization passes:

  - Constant Folding  
  Pre-computes constant operations during compile time.

  - Dead Code Elimination  
  Removes nodes that do not affect the final output.

  - Operator Fusion  
  Combines sequences like `Linear → ReLU` into a single fused operation.

• Benchmarks optimized graph execution against the original PyTorch model.

---

## Optimization Example

### Before Optimization

Input  
 ↓  
Linear  
 ↓  
ReLU  
 ↓  
Linear  
 ↓  
Output  

### After Optimization

Input  
 ↓  
LinearReLU  
 ↓  
Linear  
 ↓  
Output  

Operator fusion reduces the number of operations executed during inference.

---

## How to Run

Install dependencies
'''bash
pip install -r requirements.txt
'''

## Run the project
'''bash
python main.py
'''

## Author

**Manasa Mummadi**
Master's in Computer Science
George Mason University
