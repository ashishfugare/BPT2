# test_import.py
from crypto_problem import CryptoArithmeticProblem, OperationType

# Create a simple problem
problem = CryptoArithmeticProblem()
problem.add_constraint(["SEND", "MORE"], OperationType.ADDITION, "MONEY")
print(problem)

# Try importing other modules to verify they work
from csp_solver import CSPSolver
solver = CSPSolver(problem)
print("Imports working correctly!")