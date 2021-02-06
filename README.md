# qlib
Library of Quantum Algorithms

The idea is to create a package that will enable one to solve typical problems using a single package and be able to 
delegate compute to either a classical computer, a quantum system (Qiskit, IonQ, AWS Braket, etc), or use another type 
of accelerator such as GPU/TPU.

## Algorithms

* Linear solvers
  * Qiskit (unoptimized)
  * NumPy
  * PyTorch
  * AWS Braket support is coming...
