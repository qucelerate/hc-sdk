import numpy as np

from abc import ABC, abstractmethod

from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import HHL, NumPyLSsolver
from qiskit.aqua.components.eigs import EigsQPE
from qiskit.aqua.components.initial_states import Custom
from qiskit.aqua.components.reciprocals import LookupRotation
from qiskit.aqua.operators import MatrixOperator
from qiskit.circuit.library import QFT
from qiskit.providers import BaseBackend


class AbstractLinearSolver(ABC):
    """
    Solver for systems of linear equations.
    """

    @abstractmethod
    def solve(self, matrix, vector):
        """
        Solves equations of the form A|x〉=|b〉
        :param matrix: A - matrix corresponding to equation coefficients
        :param vector: b - vector representing the desired product
        :return: x - solution vector
        """
        raise NotImplementedError()


class QiskitLinearSolver(AbstractLinearSolver):
    """
    Package for solving linear systems of equations using Qiskit HHL algorithm. The code in this class is borrowed from
    Qiskit book and is under Apache License 2.0.
    """
    def __init__(self,
                 backend: BaseBackend,
                 num_ancillae: int = 5,
                 num_time_slices: int = 50,
                 negative_evals: bool = False):
        self.backend = backend
        self.num_ancillae = num_ancillae
        self.num_time_slices = num_time_slices
        self.negative_evals = negative_evals

    def create_eigs(self, matrix, num_ancillae, num_time_slices, negative_evals):
        ne_qfts = [None, None]
        if negative_evals:
            num_ancillae += 1
            ne_qfts = [QFT(num_ancillae - 1), QFT(num_ancillae - 1).inverse()]

        return EigsQPE(MatrixOperator(matrix=matrix),
                       QFT(num_ancillae).inverse(),
                       num_time_slices=num_time_slices,
                       num_ancillae=num_ancillae,
                       expansion_mode='suzuki',
                       expansion_order=2,
                       evo_time=np.pi*3/4,
                       negative_evals=negative_evals,
                       ne_qfts=ne_qfts)

    def solve(self, matrix, vector):
        orig_size = len(vector)
        matrix, vector, truncate_powerdim, truncate_hermitian = HHL.matrix_resize(matrix, vector)

        # Initialize eigenvalue finding module
        eigs = self.create_eigs(matrix, self.num_ancillae, self.num_time_slices, self.negative_evals)
        num_q, num_a = eigs.get_register_sizes()

        # Initialize initial state module
        init_state = Custom(num_q, state_vector=vector)

        # Initialize reciprocal rotation module
        reci = LookupRotation(negative_evals=eigs._negative_evals, evo_time=eigs._evo_time)

        algo = HHL(matrix, vector, truncate_powerdim, truncate_hermitian, eigs,
                   init_state, reci, num_q, num_a, orig_size)

        result = algo.run(QuantumInstance(self.backend))
        return result['solution']


class NumPyLinearSolver(AbstractLinearSolver):
    def solve(self, matrix, vector):
        result = NumPyLSsolver(matrix, vector).run()
        return result['solution']
