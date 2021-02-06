# qlib
Library of Quantum Algorithms

qlib is a quantum computing library for solving various math/compute problems. While the focus is on Quantum the idea is to be able to swap out various types of accelerators (Qiskit, IonQ, AWS Braket, D-Wave, etc) without changing client code. This is achieved by setting up interfaces on top of known accelerators such as quantum processing units (QPU), tensor processing units (TPU), graphical processing units (GPU), central processing units (CPU), simulators, hardware RNG, and other types of specialized hardware/software modules supporting a given mathematical method.

## Algorithms

* Linear solvers
  * Qiskit (unoptimized)
  * NumPy
  * PyTorch
  * AWS Braket support is coming...
* Scheduling
  * Heterogeneus scheduling for general purpose computing (to be started when some of the base algos are supported by qlib)

## Modules

The idea is to separate abstraction layer from implementation at a later point when a sufficient number of algorithms is implemneted (sufficient for demonstrating Heterogeneus scheduling).
