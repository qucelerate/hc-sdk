# QuCelerate Hybrid Computing SDK

QuCelerate hybrid computing software development kit aids in solving various math/compute problems. While the focus is on Quantum the idea is to be able to swap out various types of accelerators (Qiskit/IBM Quantum Experience, IonQ, AWS Braket, D-Wave, etc) without changing client code. This is achieved by setting up interfaces on top of known accelerators such as quantum processing units (QPU), tensor processing units (TPU), graphical processing units (GPU), central processing units (CPU), simulators, hardware RNG, and other types of specialized hardware/software modules supporting a given mathematical method.

## Algorithms

* Linear solvers
  * Qiskit (unoptimized)
  * NumPy
  * PyTorch
  * AWS Braket support is coming...
* Scheduling
  * Heterogeneous scheduling for general purpose computing (to be started when some of the base algos are supported by QuCelerate SDK)

## Modules

The idea is to separate abstraction layer from implementation at a later point when a sufficient number of algorithms is implemneted (sufficient for demonstrating heterogeneous scheduling).

## Draft architecture

![SDK](https://user-images.githubusercontent.com/1936580/126837204-f393d694-cbed-4da1-abf8-aec82c73da9d.png)

## Donations

## Crowd funding
