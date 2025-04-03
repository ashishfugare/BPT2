import time
import random
from typing import Dict, List, Set, Tuple, Optional, Any, Callable
from collections import deque, defaultdict

from crypto_problem import CryptoArithmeticProblem


class CSPSolver:
    """
    Base CSP solver with backtracking search.
    """
    
    def __init__(self, problem: CryptoArithmeticProblem):
        self.problem = problem
        self.domains = problem.generate_domains()
        self.assignment: Dict[str, int] = {}
        self.stats = {
            "nodes_explored": 0,
            "backtracks": 0,
            "constraint_checks": 0,
        }
    
    def solve(self, timeout: int = 300) -> Optional[Dict[str, int]]:
        """
        Solve the CSP problem with backtracking search.
        Returns the solution assignment or None if no solution exists.
        
        Args:
            timeout: Maximum time in seconds to spend searching
        """
        self.stats = {
            "nodes_explored": 0,
            "backtracks": 0,
            "constraint_checks": 0,
        }
        start_time = time.time()
        
        # Start the recursive backtracking search
        result = self._backtrack(start_time, timeout)
        
        self.stats["runtime"] = time.time() - start_time
        return result
    
    def _backtrack(self, start_time: float, timeout: int) -> Optional[Dict[str, int]]:
        """
        Recursive backtracking search to find a solution.
        """
        # Check timeout
        if time.time() - start_time > timeout:
            return None
        
        # Check if assignment is complete
        if len(self.assignment) == len(self.problem.variables):
            return self.assignment.copy()
        
        # Select an unassigned variable
        var = self._select_unassigned_variable()
        
        # Try each value in the domain
        for value in self._order_domain_values(var):
            self.stats["nodes_explored"] += 1
            
            # Check if the value is consistent with current assignment
            self.assignment[var] = value
            self.stats["constraint_checks"] += 1
            
            if self.problem.evaluate(self.assignment):
                # Recursive call with the new assignment
                result = self._backtrack(start_time, timeout)
                if result is not None:
                    return result
            
            # If we get here, the assignment failed
            del self.assignment[var]
            self.stats["backtracks"] += 1
        
        # No solution found in this branch
        return None
    
    def _select_unassigned_variable(self) -> str:
        """
        Select an unassigned variable (simple implementation).
        Override in subclasses to implement heuristics.
        """
        unassigned = [var for var in self.problem.variables if var not in self.assignment]
        return unassigned[0] if unassigned else None
    
    def _order_domain_values(self, var: str) -> List[int]:
        """
        Order domain values for a variable (simple implementation).
        Override in subclasses to implement heuristics.
        """
        return self.domains[var]


class MRVSolver(CSPSolver):
    """
    CSP Solver with Minimum Remaining Values (MRV) heuristic.
    """
    
    def _select_unassigned_variable(self) -> str:
        """
        Select the variable with the fewest remaining values in its domain.
        """
        unassigned = [var for var in self.problem.variables if var not in self.assignment]
        
        if not unassigned:
            return None
        
        # Find the variable with the minimum remaining values
        return min(unassigned, key=lambda var: len(self.domains[var]))


class ForwardCheckingSolver(MRVSolver):
    """
    CSP Solver with Forward Checking.
    """
    
    def __init__(self, problem: CryptoArithmeticProblem):
        super().__init__(problem)
        # Keep track of current domains during search
        self.current_domains = {var: list(values) for var, values in self.domains.items()}
    
    def solve(self, timeout: int = 300) -> Optional[Dict[str, int]]:
        """
        Reset current domains before solving.
        """
        self.current_domains = {var: list(values) for var, values in self.domains.items()}
        return super().solve(timeout)
    
    def _backtrack(self, start_time: float, timeout: int) -> Optional[Dict[str, int]]:
        """
        Recursive backtracking search with forward checking.
        """
        # Check timeout
        if time.time() - start_time > timeout:
            return None
        
        # Check if assignment is complete
        if len(self.assignment) == len(self.problem.variables):
            return self.assignment.copy()
        
        # Select an unassigned variable
        var = self._select_unassigned_variable()
        
        # Try each value in the domain
        for value in self._order_domain_values(var):
            self.stats["nodes_explored"] += 1
            
            # Make assignment and check if consistent
            self.assignment[var] = value
            self.stats["constraint_checks"] += 1
            
            if self.problem.evaluate(self.assignment):
                # Backup domains for restoration
                old_domains = {v: list(self.current_domains[v]) for v in self.current_domains}
                
                # Perform forward checking
                if self._forward_check(var, value):
                    # Recursive call with the new assignment
                    result = self._backtrack(start_time, timeout)
                    if result is not None:
                        return result
                
                # Restore domains
                self.current_domains = old_domains
            
            # If we get here, the assignment failed
            del self.assignment[var]
            self.stats["backtracks"] += 1
        
        # No solution found in this branch
        return None
    
    def _forward_check(self, var: str, value: int) -> bool:
        """
        Perform forward checking after assigning var = value.
        Remove inconsistent values from the domains of unassigned variables.
        Returns False if any domain becomes empty.
        """
        for constraint_idx, (left_operands, operation, right_result) in enumerate(self.problem.constraints):
            # Get all variables in this constraint
            all_words = left_operands + [right_result]
            all_vars = set()
            for word in all_words:
                all_vars.update(word)
            
            # Check if this constraint involves the assigned variable
            if var in all_vars:
                # Check each unassigned variable in this constraint
                for other_var in all_vars:
                    if other_var != var and other_var not in self.assignment:
                        # Test each value in the domain
                        domain_copy = list(self.current_domains[other_var])
                        for test_value in domain_copy:
                            test_assignment = self.assignment.copy()
                            test_assignment[other_var] = test_value
                            
                            self.stats["constraint_checks"] += 1
                            if not self.problem.evaluate(test_assignment, constraint_idx):
                                self.current_domains[other_var].remove(test_value)
                        
                        # Check if domain became empty
                        if not self.current_domains[other_var]:
                            return False
        
        return True
    
    def _select_unassigned_variable(self) -> str:
        """
        Select the variable with the fewest remaining values in its current domain.
        """
        unassigned = [var for var in self.problem.variables if var not in self.assignment]
        
        if not unassigned:
            return None
        
        # Find the variable with the minimum remaining values in current domain
        return min(unassigned, key=lambda var: len(self.current_domains[var]))
    
    def _order_domain_values(self, var: str) -> List[int]:
        """
        Return values from the current domain.
        """
        return self.current_domains[var]


class AC3Solver(ForwardCheckingSolver):
    """
    CSP Solver with AC-3 constraint propagation.
    """
    
    def solve(self, timeout: int = 300) -> Optional[Dict[str, int]]:
        """
        Run AC-3 as a preprocessing step before solving.
        """
        self.current_domains = {var: list(values) for var, values in self.domains.items()}
        
        # Run AC-3 as preprocessing
        if not self._ac3():
            # Problem is unsolvable
            return None
        
        return super().solve(timeout)
    
    def _backtrack(self, start_time: float, timeout: int) -> Optional[Dict[str, int]]:
        """
        Recursive backtracking search with AC-3.
        """
        # Check timeout
        if time.time() - start_time > timeout:
            return None
        
        # Check if assignment is complete
        if len(self.assignment) == len(self.problem.variables):
            return self.assignment.copy()
        
        # Select an unassigned variable
        var = self._select_unassigned_variable()
        
        # Try each value in the domain
        for value in self._order_domain_values(var):
            self.stats["nodes_explored"] += 1
            
            # Make assignment and check if consistent
            self.assignment[var] = value
            self.stats["constraint_checks"] += 1
            
            if self.problem.evaluate(self.assignment):
                # Backup domains for restoration
                old_domains = {v: list(self.current_domains[v]) for v in self.current_domains}
                
                # Remove value from current domain to simulate assignment
                self.current_domains[var] = [value]
                
                # Run AC-3 to propagate constraints
                if self._ac3():
                    # Recursive call with the new assignment
                    result = self._backtrack(start_time, timeout)
                    if result is not None:
                        return result
                
                # Restore domains
                self.current_domains = old_domains
            
            # If we get here, the assignment failed
            del self.assignment[var]
            self.stats["backtracks"] += 1
        
        # No solution found in this branch
        return None
    
    def _ac3(self) -> bool:
        """
        Run the AC-3 algorithm to enforce arc consistency.
        Returns False if any domain becomes empty.
        """
        # Create a queue of arcs (variable pairs)
        queue = deque()
        
        # Add all arcs to the queue
        for constraint_idx, (left_operands, operation, right_result) in enumerate(self.problem.constraints):
            # Get all variables in this constraint
            all_words = left_operands + [right_result]
            all_vars = set()
            for word in all_words:
                all_vars.update(word)
            
            # Add all variable pairs as arcs
            for var1 in all_vars:
                for var2 in all_vars:
                    if var1 != var2:
                        queue.append((var1, var2, constraint_idx))
        
        # Process all arcs
        while queue:
            var1, var2, constraint_idx = queue.popleft()
            
            # Skip if either variable is already assigned
            if var1 in self.assignment or var2 in self.assignment:
                continue
            
            # Revise the domain of var1 with respect to var2
            if self._revise(var1, var2, constraint_idx):
                # Check if domain became empty
                if not self.current_domains[var1]:
                    return False
                
                # Get all variables in this constraint
                constraint = self.problem.constraints[constraint_idx]
                all_words = constraint[0] + [constraint[2]]
                all_vars = set()
                for word in all_words:
                    all_vars.update(word)
                
                # Add new arcs for all neighbors of var1 except var2
                for var3 in all_vars:
                    if var3 != var1 and var3 != var2 and var3 not in self.assignment:
                        queue.append((var3, var1, constraint_idx))
        
        return True
    
    def _revise(self, var1: str, var2: str, constraint_idx: int) -> bool:
        """
        Revise the domain of var1 with respect to var2.
        Returns True if the domain of var1 was changed.
        """
        revised = False
        
        # Check each value in the domain of var1
        domain_copy = list(self.current_domains[var1])
        for value1 in domain_copy:
            # Check if there is a value in the domain of var2 that satisfies the constraint
            consistent = False
            
            for value2 in self.current_domains[var2]:
                # Create a test assignment
                test_assignment = self.assignment.copy()
                test_assignment[var1] = value1
                test_assignment[var2] = value2
                
                self.stats["constraint_checks"] += 1
                if self.problem.evaluate(test_assignment, constraint_idx):
                    consistent = True
                    break
            
            # If no consistent value in var2 for this value in var1, remove it
            if not consistent:
                self.current_domains[var1].remove(value1)
                revised = True
        
        return revised


class ConflictDirectedBackjumpingSolver(AC3Solver):
    """
    CSP Solver with Conflict-Directed Backjumping.
    """
    
    def __init__(self, problem: CryptoArithmeticProblem):
        super().__init__(problem)
        # Conflict set for each variable
        self.conflict_sets: Dict[str, Set[str]] = {}
    
    def solve(self, timeout: int = 300) -> Optional[Dict[str, int]]:
        """
        Reset conflict sets before solving.
        """
        self.conflict_sets = {var: set() for var in self.problem.variables}
        return super().solve(timeout)
    
    def _backtrack_with_cdb(self, var: str, start_time: float, timeout: int) -> Tuple[Optional[Dict[str, int]], Set[str]]:
        """
        Recursive backtracking search with conflict-directed backjumping.
        Returns (solution, conflict_set)
        """
        # Check timeout
        if time.time() - start_time > timeout:
            return None, set()
        
        # Check if assignment is complete
        if len(self.assignment) == len(self.problem.variables):
            return self.assignment.copy(), set()
        
        # Select an unassigned variable
        next_var = self._select_unassigned_variable()
        
        # Initialize conflict set for this variable
        self.conflict_sets[next_var] = set()
        
        # Try each value in the domain
        for value in self._order_domain_values(next_var):
            self.stats["nodes_explored"] += 1
            
            # Check if value is consistent with constraints
            self.assignment[next_var] = value
            self.stats["constraint_checks"] += 1
            
            # Check for consistency, record conflicting variables
            consistent = True
            for var_past in self.assignment:
                if var_past == next_var:
                    continue
                    
                # Create a test assignment with just these two variables
                test_assignment = {var_past: self.assignment[var_past], 
                                  next_var: value}
                
                self.stats["constraint_checks"] += 1
                if not self.problem.evaluate(test_assignment):
                    consistent = False
                    self.conflict_sets[next_var].add(var_past)
            
            if consistent:
                # Backup domains for restoration
                old_domains = {v: list(self.current_domains[v]) for v in self.current_domains}
                
                # Remove value from current domain to simulate assignment
                self.current_domains[next_var] = [value]
                
                # Run AC-3 to propagate constraints
                if self._ac3():
                    # Recursive call with the new assignment
                    result, conflict_set = self._backtrack_with_cdb(next_var, start_time, timeout)
                    
                    if result is not None:
                        return result, set()
                    
                    # If conflict doesn't involve current variable, backjump
                    if next_var not in conflict_set:
                        # Remove current variable from assignment
                        del self.assignment[next_var]
                        
                        # Restore domains
                        self.current_domains = old_domains
                        
                        # Return the conflict set from deeper level
                        self.stats["backtracks"] += 1
                        return None, conflict_set
                    
                    # Update conflict set with variables that caused this value to fail
                    self.conflict_sets[next_var].update(conflict_set - {next_var})
                
                # Restore domains
                self.current_domains = old_domains
            
            # If we get here, the assignment failed
            del self.assignment[next_var]
            self.stats["backtracks"] += 1
        
        # Return the union of all conflicts that caused this variable to fail
        return None, self.conflict_sets[next_var]
    
    def _backtrack(self, start_time: float, timeout: int) -> Optional[Dict[str, int]]:
        """
        Start the conflict-directed backjumping algorithm.
        """
        result, _ = self._backtrack_with_cdb(None, start_time, timeout)
        return result


class NogoodLearningCSPSolver(ConflictDirectedBackjumpingSolver):
    """
    CSP Solver with nogood learning (clause learning).
    """
    
    def __init__(self, problem: CryptoArithmeticProblem):
        super().__init__(problem)
        # Storage for learned nogoods (assignments that lead to failure)
        self.nogoods: List[Dict[str, int]] = []
        
    def solve(self, timeout: int = 300) -> Optional[Dict[str, int]]:
        """
        Reset nogoods before solving.
        """
        self.nogoods = []
        return super().solve(timeout)
    
    def _check_nogoods(self, assignment: Dict[str, int]) -> bool:
        """
        Check if the current assignment violates any learned nogoods.
        """
        for nogood in self.nogoods:
            # Check if the nogood is a subset of the current assignment
            if all(var in assignment and assignment[var] == val for var, val in nogood.items()):
                return False
        return True
    
    def _learn_nogood(self, conflict_set: Set[str]) -> None:
        """
        Learn a new nogood from the conflict set.
        """
        if not conflict_set:
            return
            
        # Create a minimal nogood from the conflict set
        nogood = {var: self.assignment[var] for var in conflict_set if var in self.assignment}
        
        # Only add if not already known
        if nogood and nogood not in self.nogoods:
            self.nogoods.append(nogood)
    
    def _backtrack_with_cdb(self, var: str, start_time: float, timeout: int) -> Tuple[Optional[Dict[str, int]], Set[str]]:
        """
        Recursive backtracking search with conflict-directed backjumping and nogood learning.
        """
        # Check timeout
        if time.time() - start_time > timeout:
            return None, set()
        
        # Check if assignment is complete
        if len(self.assignment) == len(self.problem.variables):
            return self.assignment.copy(), set()
        
        # Check against learned nogoods
        if not self._check_nogoods(self.assignment):
            # Find the conflict set from the violated nogood
            for nogood in self.nogoods:
                if all(var in self.assignment and self.assignment[var] == val for var, val in nogood.items()):
                    return None, set(nogood.keys())
        
        # Select an unassigned variable
        next_var = self._select_unassigned_variable()
        
        # Initialize conflict set for this variable
        self.conflict_sets[next_var] = set()
        
        # Try each value in the domain
        for value in self._order_domain_values(next_var):
            self.stats["nodes_explored"] += 1
            
            # Check if value is consistent with constraints
            self.assignment[next_var] = value
            self.stats["constraint_checks"] += 1
            
            # Check if this partial assignment violates any nogoods
            if not self._check_nogoods(self.assignment):
                continue
            
            # Check for consistency, record conflicting variables
            consistent = True
            for var_past in self.assignment:
                if var_past == next_var:
                    continue
                    
                # Create a test assignment with just these two variables
                test_assignment = {var_past: self.assignment[var_past], 
                                  next_var: value}
                
                self.stats["constraint_checks"] += 1
                if not self.problem.evaluate(test_assignment):
                    consistent = False
                    self.conflict_sets[next_var].add(var_past)
            
            if consistent:
                # Backup domains for restoration
                old_domains = {v: list(self.current_domains[v]) for v in self.current_domains}
                
                # Remove value from current domain to simulate assignment
                self.current_domains[next_var] = [value]
                
                # Run AC-3 to propagate constraints
                if self._ac3():
                    # Recursive call with the new assignment
                    result, conflict_set = self._backtrack_with_cdb(next_var, start_time, timeout)
                    
                    if result is not None:
                        return result, set()
                    
                    # Learn from this conflict
                    self._learn_nogood(conflict_set)
                    
                    # If conflict doesn't involve current variable, backjump
                    if next_var not in conflict_set:
                        # Remove current variable from assignment
                        del self.assignment[next_var]
                        
                        # Restore domains
                        self.current_domains = old_domains
                        
                        # Return the conflict set from deeper level
                        self.stats["backtracks"] += 1
                        return None, conflict_set
                    
                    # Update conflict set with variables that caused this value to fail
                    self.conflict_sets[next_var].update(conflict_set - {next_var})
                
                # Restore domains
                self.current_domains = old_domains
            
            # If we get here, the assignment failed
            del self.assignment[next_var]
            self.stats["backtracks"] += 1
        
        # Return the union of all conflicts that caused this variable to fail
        return None, self.conflict_sets[next_var]


class RandomRestartCSPSolver(NogoodLearningCSPSolver):
    """
    CSP Solver with random restarts.
    """
    
    def __init__(self, problem: CryptoArithmeticProblem, num_restarts: int = 5):
        super().__init__(problem)
        self.num_restarts = num_restarts
    
    def solve(self, timeout: int = 300) -> Optional[Dict[str, int]]:
        """
        Solve with multiple random restarts.
        """
        self.stats = {
            "nodes_explored": 0,
            "backtracks": 0,
            "constraint_checks": 0,
            "restarts": 0,
        }
        
        overall_start_time = time.time()
        remaining_time = timeout
        
        for restart in range(self.num_restarts):
            self.stats["restarts"] = restart
            
            # Reinitialize for this restart
            self.assignment = {}
            self.conflict_sets = {var: set() for var in self.problem.variables}
            self.nogoods = []
            self.current_domains = {var: list(self.domains[var]) for var in self.domains}
            
            # Randomize variable and value ordering
            self._randomize_domains()
            
            # Run AC-3 as preprocessing
            if not self._ac3():
                # Problem is unsolvable
                continue
            
            # Calculate remaining time
            elapsed = time.time() - overall_start_time
            remaining_time = timeout - elapsed
            
            if remaining_time <= 0:
                break
                
            # Start the search
            start_time = time.time()
            result = self._backtrack(start_time, remaining_time)
            
            if result is not None:
                # Solution found
                self.stats["runtime"] = time.time() - overall_start_time
                return result
        
        # No solution found after all restarts
        self.stats["runtime"] = time.time() - overall_start_time
        return None
    
    def _randomize_domains(self) -> None:
        """
        Randomize the order of values in domains to diversify search.
        """
        for var in self.current_domains:
            random.shuffle(self.current_domains[var])


# Example usage
if __name__ == "__main__":
    # Create a classic SEND + MORE = MONEY problem
    from crypto_problem import CryptoArithmeticProblem, OperationType
    
    problem = CryptoArithmeticProblem()
    problem.add_constraint(["SEND", "MORE"], OperationType.ADDITION, "MONEY")
    
    # Try different solvers
    solvers = [
        ("Basic CSP", CSPSolver(problem)),
        ("MRV Heuristic", MRVSolver(problem)),
        ("Forward Checking", ForwardCheckingSolver(problem)),
        ("AC-3", AC3Solver(problem)),
        ("Conflict-Directed Backjumping", ConflictDirectedBackjumpingSolver(problem)),
        ("Nogood Learning", NogoodLearningCSPSolver(problem)),
        ("Random Restart", RandomRestartCSPSolver(problem, num_restarts=3))
    ]
    
    # Run each solver
    for name, solver in solvers:
        print(f"\nRunning {name}...")
        solution = solver.solve(timeout=60)
        
        if solution:
            print(f"Solution found: {solution}")
        else:
            print("No solution found.")
            
        print(f"Statistics: {solver.stats}")