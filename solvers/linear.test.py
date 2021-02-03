import os
import unittest

import numpy as np
import numpy.testing as npt

from qiskit import IBMQ, Aer
from qiskit.quantum_info import state_fidelity

from solvers.linear import QiskitLinearSolver, NumPyLinearSolver


def get_fidelity(hhl, ref):
    solution_hhl_normed = hhl / np.linalg.norm(hhl)
    solution_ref_normed = ref / np.linalg.norm(ref)
    return state_fidelity(solution_hhl_normed, solution_ref_normed)


class QiskitLinearSolverTest(unittest.TestCase):
    def setUp(self):
        self.matrix = [[1, -1/3], [-1/3, 1]]
        self.vector = [1, 0]

    def test_solve_simulator(self):
        simulator_backend = Aer.get_backend('statevector_simulator')
        qiskit_solver = QiskitLinearSolver(simulator_backend)
        numpy_sovler = NumPyLinearSolver()
        quantum_result = qiskit_solver.solve(self.matrix, self.vector)
        classical_result = numpy_sovler.solve(self.matrix, self.vector)
        fidelity = get_fidelity(quantum_result, classical_result)
        npt.assert_almost_equal(fidelity, 1.0)

    @unittest.skip("Requires API key")
    def test_solve_ibm_quantum_experience(self):
        IBMQ.save_account(os.getenv('QE_API_KEY'), overwrite=True)
        provider = IBMQ.load_account()
        backend = provider.get_backend('ibmq_16_melbourne')
        qiskit_solver = QiskitLinearSolver(backend)
        numpy_sovler = NumPyLinearSolver()
        quantum_result = qiskit_solver.solve(self.matrix, self.vector)
        classical_result = numpy_sovler.solve(self.matrix, self.vector)
        fidelity = get_fidelity(quantum_result, classical_result)
        npt.assert_almost_equal(fidelity, 1.0)


if __name__ == '__main__':
    unittest.main()
