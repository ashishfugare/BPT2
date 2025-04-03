import time
import itertools
from typing import Dict, List, Set, Tuple, Optional, Any

# For Z3 SAT solver
try:
    from z3 import *
except ImportError:
    print("Z3 solver not available. Please install with: pip install z3-solver")

# For PySAT (which includes MiniSAT, Glucose, and other SAT solvers)
try:
    from pysat.formula import CNF
    from pysat.solvers import Solver as PySATSolver
except ImportError:
    print("PySAT not available. Please install with: pip install python-sat")

from crypto_problem import CryptoArithmeticProblem, OperationType


class SATSolverBase:
    """
    Base class for SAT solver integration with cryptoarithmetic problems.
    """
    
    def __init__(self, problem: CryptoArithmeticProblem):
        self.problem = problem
        self.stats = {
            "encoding_time": 0,
            "solving_time": 0,
            "variables_count": 0,
            "clauses_count": 0,
        }
    
    def solve(self, timeout: int = 300) -> Optional[Dict[str, int]]:
        """
        Encode the problem as SAT and solve it.
        Returns the solution assignment or None if no solution exists.
        
        Args:
            timeout: Maximum time in seconds to spend solving
        """
        start_encoding = time.time()
        self._encode_problem()
        self.stats["encoding_time"] = time.time() - start_encoding
        
        start_solving = time.time()
        solution = self._solve_sat(timeout)
        self.stats["solving_time"] = time.time() - start_solving
        
        self.stats["total_time"] = self.stats["encoding_time"] + self.stats["solving_time"]
        
        return solution
    
    def _encode_problem(self) -> None:
        """
        Encode the cryptoarithmetic problem as SAT.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _solve_sat(self, timeout: int) -> Optional[Dict[str, int]]:
        """
        Solve the SAT problem.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")


class Z3SATSolver(SATSolverBase):
    """
    SAT solver implementation using Z3.
    """
    
    def __init__(self, problem: CryptoArithmeticProblem):
        super().__init__(problem)
        self.solver = None
        self.variables = {}
    
    def _encode_problem(self) -> None:
        """
        Encode the cryptoarithmetic problem using Z3.
        """
        self.solver = Solver()
        
        # Create Z3 integer variables for each letter
        self.variables = {}
        for var in self.problem.variables:
            self.variables[var] = Int(var)
            
            # Domain constraints: value is in [0, base-1]
            self.solver.add(self.variables[var] >= 0)
            self.solver.add(self.variables[var] < self.problem.base)
            
            # First letter constraints: cannot be 0
            if var in self.problem.first_letter_constraints:
                self.solver.add(self.variables[var] > 0)
        
        # All variables must have different values (if base allows it)
        if len(self.problem.variables) <= self.problem.base:
            self.solver.add(Distinct([self.variables[var] for var in self.problem.variables]))
        
        # Add constraints for each cryptarithmetic equation
        for left_operands, operation, right_result in self.problem.constraints:
            # Create expressions for left operands
            left_exprs = []
            for word in left_operands:
                expr = 0
                for i, letter in enumerate(reversed(word)):
                    expr += self.variables[letter] * (self.problem.base ** i)
                left_exprs.append(expr)
            
            # Create expression for right result
            right_expr = 0
            for i, letter in enumerate(reversed(right_result)):
                right_expr += self.variables[letter] * (self.problem.base ** i)
            
            # Add constraint based on operation
            if operation == OperationType.ADDITION:
                self.solver.add(Sum(left_exprs) == right_expr)
            elif operation == OperationType.SUBTRACTION:
                self.solver.add(left_exprs[0] - Sum(left_exprs[1:]) == right_expr)
            elif operation == OperationType.MULTIPLICATION:
                product = left_exprs[0]
                for expr in left_exprs[1:]:
                    product *= expr
                self.solver.add(product == right_expr)
            elif operation == OperationType.DIVISION:
                quotient = left_exprs[0]
                for expr in left_exprs[1:]:
                    # Add constraint to avoid division by zero
                    self.solver.add(expr != 0)
                    # In Z3, we use multiplication to represent division
                    # (since we're working with integers)
                    self.solver.add(quotient == right_expr * expr)
                    quotient = right_expr * expr
        
        # Count variables and constraints
        self.stats["variables_count"] = len(self.problem.variables)
        self.stats["clauses_count"] = len(self.solver.assertions())
    
    def _solve_sat(self, timeout: int) -> Optional[Dict[str, int]]:
        """
        Solve using Z3.
        """
        if self.solver is None:
            raise ValueError("Problem not encoded yet. Call _encode_problem first.")
        
        # Set timeout
        self.solver.set("timeout", timeout * 1000)  # Z3 timeout is in milliseconds
        
        # Check if satisfiable
        result = self.solver.check()
        
        if result == sat:
            # Get model
            model = self.solver.model()
            
            # Extract solution
            solution = {}
            for var in self.problem.variables:
                solution[var] = model[self.variables[var]].as_long()
            
            return solution
        else:
            return None


class MiniSATSolver(SATSolverBase):
    """
    SAT solver implementation using PySAT with MiniSAT.
    """
    
    def __init__(self, problem: CryptoArithmeticProblem):
        super().__init__(problem)
        self.cnf = None
        self.var_mapping = {}  # Maps (letter, value) to SAT variable
        self.reverse_mapping = {}  # Maps SAT variable to (letter, value)
    
    def _encode_problem(self) -> None:
        """
        Encode the cryptoarithmetic problem as CNF for MiniSAT.
        This encoding uses the direct encoding approach.
        """
        self.cnf = CNF()
        
        # Create variable mapping
        var_id = 1  # SAT variables start at 1
        self.var_mapping = {}
        self.reverse_mapping = {}
        
        for letter in self.problem.variables:
            for value in range(self.problem.base):
                # Skip 0 for first letters
                if value == 0 and letter in self.problem.first_letter_constraints:
                    continue
                
                self.var_mapping[(letter, value)] = var_id
                self.reverse_mapping[var_id] = (letter, value)
                var_id += 1
        
        # At least one value per letter
        for letter in self.problem.variables:
            values = range(self.problem.base)
            if letter in self.problem.first_letter_constraints:
                values = range(1, self.problem.base)
                
            clause = [self.var_mapping[(letter, val)] for val in values]
            self.cnf.append(clause)
        
        # At most one value per letter
        for letter in self.problem.variables:
            values = range(self.problem.base)
            if letter in self.problem.first_letter_constraints:
                values = range(1, self.problem.base)
                
            for val1, val2 in itertools.combinations(values, 2):
                self.cnf.append([-self.var_mapping[(letter, val1)], -self.var_mapping[(letter, val2)]])
        
        # All letters must have different values
        if len(self.problem.variables) <= self.problem.base:
            for letter1, letter2 in itertools.combinations(self.problem.variables, 2):
                for val in range(self.problem.base):
                    # Skip 0 for first letters
                    if val == 0 and (letter1 in self.problem.first_letter_constraints or 
                                     letter2 in self.problem.first_letter_constraints):
                        continue
                    
                    # Both letters can't have the same value
                    if (letter1, val) in self.var_mapping and (letter2, val) in self.var_mapping:
                        self.cnf.append([-self.var_mapping[(letter1, val)], -self.var_mapping[(letter2, val)]])
        
        # Add arithmetic constraints (this is complex - using helper methods)
        self._encode_arithmetic_constraints()
        
        # Count variables and clauses
        self.stats["variables_count"] = var_id - 1
        self.stats["clauses_count"] = len(self.cnf.clauses)
    
    def _encode_arithmetic_constraints(self) -> None:
        """
        Encode arithmetic constraints using binary adder circuits.
        This is a simplified version - full implementation would be much more complex.
        """
        # For each constraint, we need to encode the arithmetic operation
        for constraint_idx, (left_operands, operation, right_result) in enumerate(self.problem.constraints):
            # For now, only implementing addition
            if operation == OperationType.ADDITION:
                # Convert words to columns of digits
                max_len = max([len(word) for word in left_operands + [right_result]])
                
                # Create columns of letters from right to left (least to most significant)
                columns = []
                for i in range(max_len):
                    column = []
                    for word in left_operands:
                        if i < len(word):
                            column.append(word[-(i+1)])  # Take from the end
                    
                    result_letter = None
                    if i < len(right_result):
                        result_letter = right_result[-(i+1)]
                        
                    columns.append((column, result_letter))
                
                # Encode each column with carry
                carry_in = None  # No carry-in for the rightmost column
                
                for i, (column_letters, result_letter) in enumerate(columns):
                    # Create carry-out variables if needed
                    carry_out = None
                    if i < len(columns) - 1:
                        carry_out = f"carry_{constraint_idx}_{i}"
                        # Add carry_out variable to the mappings
                        for val in range(2):  # Carry is binary
                            self.var_mapping[(carry_out, val)] = len(self.reverse_mapping) + 1
                            self.reverse_mapping[len(self.reverse_mapping) + 1] = (carry_out, val)
                    
                    # Encode column sum constraint
                    self._encode_column_sum(column_letters, result_letter, carry_in, carry_out)
                    
                    # Next column's carry-in is this column's carry-out
                    carry_in = carry_out
    
    def _encode_column_sum(self, column_letters: List[str], result_letter: Optional[str], 
                          carry_in: Optional[str], carry_out: Optional[str]) -> None:
        """
        Encode constraints for a single column in an addition problem.
        This is a simplified encoding that could be further optimized.
        """
        # For each combination of values for the letters and carries
        base = self.problem.base
        
        # Generate all possible value assignments for column letters
        column_values = list(itertools.product(range(base), repeat=len(column_letters)))
        
        # Consider carry-in values
        carry_in_values = [None]
        if carry_in is not None:
            carry_in_values = [0, 1]
        
        # For each combination, check if it's valid
        for letter_vals in column_values:
            for cin in carry_in_values:
                # Calculate sum
                column_sum = sum(letter_vals)
                if cin is not None:
                    column_sum += cin
                
                # Calculate result digit and carry-out
                result_digit = column_sum % base
                cout = column_sum // base
                
                # Build clause for this combination
                clause = []
                
                # Add negated letter variables
                for letter, val in zip(column_letters, letter_vals):
                    clause.append(-self.var_mapping[(letter, val)])
                
                # Add negated carry-in variable
                if cin is not None:
                    clause.append(-self.var_mapping[(carry_in, cin)])
                
                # Add implied result variable
                if result_letter is not None:
                    clause.append(self.var_mapping[(result_letter, result_digit)])
                elif result_digit != 0:
                    # If no result letter but sum isn't 0, this combination is invalid
                    self.cnf.append(clause)
                    continue
                
                # Add implied carry-out variable
                if carry_out is not None:
                    clause.append(self.var_mapping[(carry_out, cout)])
                elif cout != 0:
                    # If no carry-out but there is a carry, this combination is invalid
                    self.cnf.append(clause)
                    continue
                
                # Add clause
                self.cnf.append(clause)
    
    def _solve_sat(self, timeout: int) -> Optional[Dict[str, int]]:
        """
        Solve using MiniSAT.
        """
        if self.cnf is None:
            raise ValueError("Problem not encoded yet. Call _encode_problem first.")
        
        # Create solver
        solver = PySATSolver(name="minisat22", bootstrap_with=self.cnf)
        
        # Set timeout (if supported)
        solver.set_timeout(timeout)
        
        # Solve
        satisfiable = solver.solve()
        
        if satisfiable:
            # Get model
            model = solver.get_model()
            
            # Extract solution
            solution = {}
            for var in model:
                if var > 0:  # Only positive literals represent assignments
                    letter, value = self.reverse_mapping[var]
                    if letter in self.problem.variables:  # Skip auxiliary variables
                        solution[letter] = value
            
            solver.delete()
            return solution
        else:
            solver.delete()
            return None


# Factory function to create appropriate SAT solver
def create_sat_solver(problem: CryptoArithmeticProblem, solver_type: str = "z3") -> SATSolverBase:
    """
    Create SAT solver of the specified type.
    
    Args:
        problem: Cryptoarithmetic problem to solve
        solver_type: Type of SAT solver to use ("z3", "minisat", etc.)
    
    Returns:
        SAT solver instance
    """
    solver_type = solver_type.lower()
    
    if solver_type == "z3":
        return Z3SATSolver(problem)
    elif solver_type == "minisat":
        return MiniSATSolver(problem)
    else:
        raise ValueError(f"Unknown SAT solver type: {solver_type}")


# Example usage
if __name__ == "__main__":
    # Create a classic SEND + MORE = MONEY problem
    from crypto_problem import CryptoArithmeticProblem, OperationType
    
    problem = CryptoArithmeticProblem()
    problem.add_constraint(["SEND", "MORE"], OperationType.ADDITION, "MONEY")
    
    # Try different SAT solvers
    sat_solvers = [
        ("Z3", create_sat_solver(problem, "z3")),
        ("MiniSAT", create_sat_solver(problem, "minisat"))
    ]
    
    # Run each solver
    for name, solver in sat_solvers:
        print(f"\nRunning {name}...")
        solution = solver.solve(timeout=60)
        
        if solution:
            print(f"Solution found: {solution}")
        else:
            print("No solution found.")
            
        print(f"Statistics: {solver.stats}")