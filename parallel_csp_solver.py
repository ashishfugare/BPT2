import time
import random
import multiprocessing
from multiprocessing import Queue, Process, Manager
from typing import Dict, List, Set, Tuple, Optional, Any, Callable
import queue
import copy
import signal

from crypto_problem import CryptoArithmeticProblem
from csp_solver import AC3Solver, NogoodLearningCSPSolver


class ParallelCSPSolver:
    """
    Base class for parallel CSP solvers.
    """
    
    def __init__(self, problem: CryptoArithmeticProblem, num_workers: int = None):
        self.problem = problem
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.stats = {
            "nodes_explored": 0,
            "backtracks": 0,
            "constraint_checks": 0,
            "runtime": 0,
            "workers_used": self.num_workers,
        }
    
    def solve(self, timeout: int = 300) -> Optional[Dict[str, int]]:
        """
        Solve using parallel processing.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")


class PortfolioParallelSolver(ParallelCSPSolver):
    """
    Portfolio-based parallel solver that runs different solver
    configurations in parallel and returns the first solution found.
    """
    
    def __init__(self, problem: CryptoArithmeticProblem, num_workers: int = None):
        super().__init__(problem, num_workers)
        
        # Define solver portfolio
        self.portfolio = [
            # Configuration: (solver_class, kwargs)
            (AC3Solver, {}),
            (NogoodLearningCSPSolver, {}),
            (AC3Solver, {"random_restarts": True}),
            (NogoodLearningCSPSolver, {"random_restarts": True}),
        ]
        
        # Ensure we don't have more workers than portfolio configurations
        self.num_workers = min(self.num_workers, len(self.portfolio))
        self.stats["workers_used"] = self.num_workers
    
    def worker_process(self, solver_class, solver_kwargs, problem, 
                      result_queue, stop_event, worker_id):
        """
        Worker process function that runs a solver configuration.
        
        Args:
            solver_class: Class of the solver to instantiate
            solver_kwargs: Keyword arguments for solver instantiation
            problem: Problem to solve
            result_queue: Queue to put results
            stop_event: Event to signal when to stop
            worker_id: ID of the worker for stats reporting
        """
        try:
            # Create a deep copy of the problem to avoid shared state issues
            problem_copy = copy.deepcopy(problem)
            
            # Create solver instance
            random_restarts = solver_kwargs.pop("random_restarts", False)
            
            if random_restarts:
                # Add randomness to the solver
                if solver_class == AC3Solver:
                    from csp_solver import RandomRestartCSPSolver
                    solver = RandomRestartCSPSolver(problem_copy, num_restarts=3)
                else:
                    solver = solver_class(problem_copy, **solver_kwargs)
                    # Randomize the domain values
                    for var in solver.domains:
                        values = list(solver.domains[var])
                        random.shuffle(values)
                        solver.domains[var] = values
            else:
                solver = solver_class(problem_copy, **solver_kwargs)
            
            # Set timeout handler
            def timeout_handler(signum, frame):
                raise TimeoutError("Solver timeout")
            
            # Register timeout handler
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            try:
                # Solve the problem
                start_time = time.time()
                solution = solver.solve(timeout=timeout)
                solve_time = time.time() - start_time
                
                # Cancel the alarm
                signal.alarm(0)
                
                if solution and not stop_event.is_set():
                    # Put solution in the queue
                    result = {
                        "solution": solution,
                        "worker_id": worker_id,
                        "solver_class": solver_class.__name__,
                        "random_restarts": random_restarts,
                        "solve_time": solve_time,
                    }
                    
                    # Include solver stats if available
                    if hasattr(solver, "stats"):
                        for key, value in solver.stats.items():
                            result[key] = value
                    
                    # Put result in queue and set stop event
                    result_queue.put(result)
                    stop_event.set()
            
            except TimeoutError:
                # Solver timed out
                pass
            
            finally:
                # Cancel the alarm in case of any exception
                signal.alarm(0)
        
        except Exception as e:
            # Log error and continue
            print(f"Worker {worker_id} error: {e}")
    
    def solve(self, timeout: int = 300) -> Optional[Dict[str, int]]:
        """
        Solve using a portfolio of solvers in parallel.
        
        Args:
            timeout: Maximum time in seconds to spend solving
            
        Returns:
            Solution assignment or None if no solution found
        """
        # Set up multiprocessing resources
        manager = Manager()
        result_queue = manager.Queue()
        stop_event = manager.Event()
        
        # Start worker processes
        workers = []
        for i in range(self.num_workers):
            # Get solver configuration from portfolio
            solver_class, solver_kwargs = self.portfolio[i % len(self.portfolio)]
            
            # Create and start worker process
            p = Process(
                target=self.worker_process,
                args=(solver_class, solver_kwargs, self.problem, result_queue, stop_event, i)
            )
            workers.append(p)
            p.start()
        
        # Wait for result or timeout
        start_time = time.time()
        solution = None
        
        try:
            # Wait for first result
            while time.time() - start_time < timeout:
                try:
                    # Try to get result from queue with timeout
                    result = result_queue.get(timeout=1)
                    solution = result.pop("solution")
                    
                    # Update stats with solver stats
                    self.stats.update(result)
                    break
                
                except queue.Empty:
                    # Check if all workers are done
                    if all(not p.is_alive() for p in workers):
                        break
        
        finally:
            # Signal all workers to stop
            stop_event.set()
            
            # Join all worker processes (with timeout)
            for p in workers:
                p.join(timeout=1)
                
                # If process is still alive, terminate it
                if p.is_alive():
                    p.terminate()
        
        # Update runtime stat
        self.stats["runtime"] = time.time() - start_time
        
        return solution


class WorkStealingParallelSolver(ParallelCSPSolver):
    """
    Work-stealing parallel solver that divides the search space and
    enables idle workers to steal work from busy ones.
    """
    
    def __init__(self, problem: CryptoArithmeticProblem, num_workers: int = None):
        super().__init__(problem, num_workers)
        self.base_solver_class = AC3Solver
    
    def worker_process(self, problem, initial_assignment, worker_queue, 
                      result_queue, shared_nogoods, stop_event, worker_id):
        """
        Worker process function for work-stealing parallel solver.
        
        Args:
            problem: Problem to solve
            initial_assignment: Initial partial assignment for this worker
            worker_queue: Queue for this worker to receive stolen work
            result_queue: Queue to put results
            shared_nogoods: Shared list of nogoods between workers
            stop_event: Event to signal when to stop
            worker_id: ID of the worker for stats reporting
        """
        try:
            # Create solver instance
            solver = self.base_solver_class(problem)
            
            # Set initial assignment
            solver.assignment = initial_assignment.copy() if initial_assignment else {}
            
            # Initialize local stats
            local_stats = {
                "nodes_explored": 0,
                "backtracks": 0,
                "constraint_checks": 0,
            }
            
            # Search stack (for DFS)
            search_stack = []
            
            # Initial state: unassigned variables and their domains
            unassigned = [var for var in problem.variables if var not in solver.assignment]
            
            if unassigned:
                # Start with first unassigned variable
                var = unassigned[0]
                for value in solver.domains[var]:
                    # Create state for each value
                    search_stack.append({
                        "var": var,
                        "value": value,
                        "assignment": solver.assignment.copy()
                    })
            
            start_time = time.time()
            
            # Main search loop
            while search_stack and not stop_event.is_set():
                # Check if timeout reached
                if time.time() - start_time > timeout:
                    break
                
                # Check if there's stolen work in the queue
                try:
                    stolen_work = worker_queue.get_nowait()
                    search_stack.extend(stolen_work)
                except queue.Empty:
                    pass
                
                # Get next state from stack
                state = search_stack.pop()
                var = state["var"]
                value = state["value"]
                solver.assignment = state["assignment"].copy()
                
                # Try assigning this value
                solver.assignment[var] = value
                local_stats["nodes_explored"] += 1
                local_stats["constraint_checks"] += 1
                
                # Check if assignment is consistent
                if problem.evaluate(solver.assignment):
                    # Check if assignment is complete
                    if len(solver.assignment) == len(problem.variables):
                        # Solution found
                        result = {
                            "solution": solver.assignment.copy(),
                            "worker_id": worker_id,
                            "solve_time": time.time() - start_time,
                        }
                        
                        # Include local stats
                        for key, value in local_stats.items():
                            result[key] = value
                        
                        # Put result in queue and set stop event
                        result_queue.put(result)
                        stop_event.set()
                        break
                    
                    # Get next unassigned variable
                    unassigned = [var for var in problem.variables if var not in solver.assignment]
                    next_var = unassigned[0]
                    
                    # Expand with new states for next variable
                    new_states = []
                    for next_value in solver.domains[next_var]:
                        new_states.append({
                            "var": next_var,
                            "value": next_value,
                            "assignment": solver.assignment.copy()
                        })
                    
                    # Add new states to stack
                    search_stack.extend(new_states)
                    
                    # If stack is large, offer to share work
                    if len(search_stack) > 10:
                        # Offer half of our work to be stolen
                        half_size = len(search_stack) // 2
                        work_to_share = search_stack[-half_size:]
                        search_stack = search_stack[:-half_size]
                        
                        # Put work in other workers' queues
                        for i in range(self.num_workers):
                            if i != worker_id:
                                try:
                                    worker_queues[i].put_nowait(work_to_share)
                                    break  # Only share with one worker
                                except queue.Full:
                                    continue
                else:
                    # Inconsistent assignment, backtrack
                    local_stats["backtracks"] += 1
            
            # If search_stack is empty and no solution found, worker is done
            if not search_stack and not stop_event.is_set():
                # Put "no solution" result in queue
                result = {
                    "solution": None,
                    "worker_id": worker_id,
                    "solve_time": time.time() - start_time,
                }
                
                # Include local stats
                for key, value in local_stats.items():
                    result[key] = value
                
                result_queue.put(result)
        
        except Exception as e:
            # Log error and continue
            print(f"Worker {worker_id} error: {e}")
    
    def solve(self, timeout: int = 300) -> Optional[Dict[str, int]]:
        """
        Solve using work-stealing parallel approach.
        
        Args:
            timeout: Maximum time in seconds to spend solving
            
        Returns:
            Solution assignment or None if no solution found
        """
        # Set up multiprocessing resources
        manager = Manager()
        result_queue = manager.Queue()
        shared_nogoods = manager.list()
        stop_event = manager.Event()
        
        # Create a queue for each worker
        global worker_queues
        worker_queues = [manager.Queue(maxsize=100) for _ in range(self.num_workers)]
        
        # Divide initial work among workers
        initial_assignments = self._divide_initial_work()
        
        # Start worker processes
        workers = []
        for i in range(self.num_workers):
            p = Process(
                target=self.worker_process,
                args=(
                    self.problem,
                    initial_assignments[i],
                    worker_queues[i],
                    result_queue,
                    shared_nogoods,
                    stop_event,
                    i
                )
            )
            workers.append(p)
            p.start()
        
        # Wait for result or timeout
        start_time = time.time()
        solution = None
        all_worker_stats = []
        
        try:
            # Continue until solution found, all workers done, or timeout
            while time.time() - start_time < timeout:
                try:
                    # Try to get result from queue with timeout
                    result = result_queue.get(timeout=1)
                    
                    # Store worker stats
                    worker_stats = {k: v for k, v in result.items() if k != "solution"}
                    all_worker_stats.append(worker_stats)
                    
                    # If solution found, stop all workers
                    if result["solution"] is not None:
                        solution = result["solution"]
                        stop_event.set()
                        break
                    
                    # If all workers have reported no solution, we're done
                    if len(all_worker_stats) == self.num_workers:
                        break
                
                except queue.Empty:
                    # Check if all workers are done
                    if all(not p.is_alive() for p in workers):
                        break
        
        finally:
            # Signal all workers to stop
            stop_event.set()
            
            # Join all worker processes (with timeout)
            for p in workers:
                p.join(timeout=1)
                
                # If process is still alive, terminate it
                if p.is_alive():
                    p.terminate()
        
        # Combine stats from all workers
        for stat_key in ["nodes_explored", "backtracks", "constraint_checks"]:
            self.stats[stat_key] = sum(worker["nodes_explored"] for worker in all_worker_stats 
                                     if stat_key in worker)
        
        # Update runtime stat
        self.stats["runtime"] = time.time() - start_time
        
        return solution
    
    def _divide_initial_work(self) -> List[Dict[str, int]]:
        """
        Divide the initial search space among workers.
        
        Returns:
            List of initial partial assignments for each worker
        """
        initial_assignments = [None] * self.num_workers
        
        # If we have only a few workers, divide based on first variable
        if self.num_workers <= self.problem.base:
            # Choose first variable to branch on
            first_var = next(iter(self.problem.variables))
            
            # Create a partial assignment for each value of first variable
            assignments = []
            for value in range(self.problem.base):
                # Skip 0 for first letters
                if value == 0 and first_var in self.problem.first_letter_constraints:
                    continue
                
                assignments.append({first_var: value})
            
            # Distribute assignments among workers
            for i, assignment in enumerate(assignments):
                if i < self.num_workers:
                    initial_assignments[i] = assignment
        
        else:
            # For many workers, we can branch on multiple variables
            # This is more complex and could be implemented as needed
            pass
        
        return initial_assignments


# Example usage
if __name__ == "__main__":
    # Create a classic SEND + MORE = MONEY problem
    from crypto_problem import CryptoArithmeticProblem, OperationType
    
    problem = CryptoArithmeticProblem()
    problem.add_constraint(["SEND", "MORE"], OperationType.ADDITION, "MONEY")
    
    # Compare sequential vs parallel solvers
    solvers = [
        ("Sequential AC3", AC3Solver(problem)),
        ("Portfolio Parallel", PortfolioParallelSolver(problem, num_workers=4)),
        ("Work-Stealing Parallel", WorkStealingParallelSolver(problem, num_workers=4))
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