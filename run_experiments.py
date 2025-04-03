import time
import os
from crypto_problem import CryptoArithmeticProblem, OperationType, ProblemGenerator
from csp_solver import (
    CSPSolver, MRVSolver, ForwardCheckingSolver, AC3Solver,
    ConflictDirectedBackjumpingSolver, NogoodLearningCSPSolver, RandomRestartCSPSolver
)
from sat_solver import create_sat_solver
from benchmarking_framework import BenchmarkRunner
from phase_transition_analyzer import PhaseTransitionAnalyzer
from parallel_csp_solver import PortfolioParallelSolver, WorkStealingParallelSolver

# Create output directories
os.makedirs("results", exist_ok=True)
os.makedirs("results/charts", exist_ok=True)
os.makedirs("results/data", exist_ok=True)

def run_comparative_experiment():
    """
    Run a comparison between CSP and SAT approaches on various problem sizes
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: CSP vs SAT Comparison")
    print("="*80)
    
    benchmark = BenchmarkRunner(output_dir="results/csp_vs_sat")
    
    # Generate problems with increasing complexity
    print("Generating benchmark problems...")
    
    problems = []
    
    # Base-10 problems (small to medium)
    for var_count in [6, 8, 10]:
        generator = ProblemGenerator(
            base=10,
            max_word_length=var_count//2,
            variable_count=var_count
        )
        # Simple addition
        problems.append(generator.generate_simple_addition())
        # Multi-operation
        problems.append(generator.generate_multi_operation(op_count=2))
    
    # Base-16 problems (medium)
    for var_count in [8, 10, 12]:
        generator = ProblemGenerator(
            base=16,
            max_word_length=var_count//2,
            variable_count=var_count
        )
        problems.append(generator.generate_simple_addition())
        problems.append(generator.generate_multi_constraint(constraint_count=2))
    
    # Define solvers for comparison
    solvers = [
        "csp_basic",          # Basic CSP (baseline)
        "csp_mrv",            # With MRV heuristic
        "csp_forward_checking", # With forward checking
        "csp_ac3",            # With AC-3
        "csp_backjumping",    # With conflict-directed backjumping
        "csp_nogood",         # With nogood learning
        "sat_z3",             # Z3 SAT solver
        "sat_minisat"         # MiniSAT solver
    ]
    
    # Run the benchmark
    print(f"Running benchmark with {len(problems)} problems and {len(solvers)} solvers...")
    results = benchmark.run_benchmark(
        problems=problems,
        solver_names=solvers,
        timeout=120,  # 2 minutes per problem
        parallel=True  # Run in parallel for speed
    )
    
    # Analyze results
    print("Analyzing results...")
    benchmark.analyze_results(results)
    
    return results

def run_phase_transition_experiment():
    """
    Analyze phase transitions in problem difficulty
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: Phase Transition Analysis")
    print("="*80)
    
    analyzer = PhaseTransitionAnalyzer(output_dir="results/phase_transition")
    
    # Analyze variable count phase transition
    print("Analyzing phase transition with variable count...")
    problems_variable = analyzer.generate_controlled_difficulty_problems(
        num_problems=5,  # 5 problems per parameter value
        param_name="variable_count",
        param_values=list(range(5, 15)),  # From 5 to 14 variables
        base=10
    )
    
    results_variable = analyzer.analyze_phase_transition(
        problems=problems_variable,
        solver_names=["csp_ac3", "sat_z3"],  # One CSP and one SAT solver
        timeout=180  # 3 minutes per problem
    )
    
    analyzer.plot_phase_transition(results_variable)
    
    # Analyze base size phase transition
    print("Analyzing phase transition with base size...")
    problems_base = analyzer.generate_controlled_difficulty_problems(
        num_problems=5,  # 5 problems per parameter value
        param_name="base",
        param_values=[4, 6, 8, 10, 12, 16, 20, 26, 32],
        base=None  # Will be set by param_values
    )
    
    results_base = analyzer.analyze_phase_transition(
        problems=problems_base,
        solver_names=["csp_ac3", "sat_z3"],  # One CSP and one SAT solver
        timeout=180  # 3 minutes per problem
    )
    
    analyzer.plot_phase_transition(results_base)
    
    return results_variable, results_base

def run_optimization_experiment():
    """
    Compare different optimization techniques on hard problems
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: Optimization Technique Comparison")
    print("="*80)
    
    # Create a specific hard problem near the phase transition
    generator = ProblemGenerator(base=16, max_word_length=6, variable_count=12)
    problem = generator.generate_multi_constraint(constraint_count=3)
    
    print(f"Created hard problem: {problem}")
    print(f"Variables: {len(problem.variables)}, Constraints: {len(problem.constraints)}")
    
    # Define solvers with increasing optimizations
    solvers = [
        ("Basic CSP", CSPSolver(problem)),
        ("MRV Heuristic", MRVSolver(problem)),
        ("Forward Checking", ForwardCheckingSolver(problem)),
        ("AC-3", AC3Solver(problem)),
        ("Conflict-Directed Backjumping", ConflictDirectedBackjumpingSolver(problem)),
        ("Nogood Learning", NogoodLearningCSPSolver(problem)),
        ("Random Restart", RandomRestartCSPSolver(problem, num_restarts=3))
    ]
    
    # Compare all solvers
    results = []
    for name, solver in solvers:
        print(f"Running {name}...")
        start_time = time.time()
        solution = solver.solve(timeout=300)  # 5 minutes
        solve_time = time.time() - start_time
        
        result = {
            "solver": name,
            "solution_found": solution is not None,
            "solution_time": solve_time,
        }
        
        # Add solver-specific stats
        if hasattr(solver, "stats"):
            for key, value in solver.stats.items():
                result[key] = value
        
        results.append(result)
        
        print(f"  Solution found: {solution is not None}")
        print(f"  Time: {solve_time:.2f} seconds")
        if hasattr(solver, "stats"):
            print(f"  Nodes explored: {solver.stats.get('nodes_explored', 'N/A')}")
            print(f"  Backtracks: {solver.stats.get('backtracks', 'N/A')}")
    
    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("results/data/optimization_comparison.csv", index=False)
    
    return results

def run_parallel_experiment():
    """
    Compare sequential vs parallel solvers
    """
    print("\n" + "="*80)
    print("EXPERIMENT 4: Parallel Solver Performance")
    print("="*80)
    
    # Create a hard problem
    generator = ProblemGenerator(base=16, max_word_length=5, variable_count=10)
    problem = generator.generate_multi_operation(op_count=2)
    
    print(f"Created problem for parallel solving: {problem}")
    
    # Compare sequential vs parallel
    solvers = [
        ("Sequential AC3", AC3Solver(problem)),
        ("Portfolio (2 workers)", PortfolioParallelSolver(problem, num_workers=2)),
        ("Portfolio (4 workers)", PortfolioParallelSolver(problem, num_workers=4)),
        ("Work-Stealing (2 workers)", WorkStealingParallelSolver(problem, num_workers=2)),
        ("Work-Stealing (4 workers)", WorkStealingParallelSolver(problem, num_workers=4))
    ]
    
    # Run each solver
    results = []
    for name, solver in solvers:
        print(f"Running {name}...")
        start_time = time.time()
        solution = solver.solve(timeout=300)  # 5 minutes
        solve_time = time.time() - start_time
        
        result = {
            "solver": name,
            "solution_found": solution is not None,
            "solution_time": solve_time
        }
        
        # Add solver-specific stats
        if hasattr(solver, "stats"):
            for key, value in solver.stats.items():
                result[key] = value
        
        results.append(result)
        
        print(f"  Solution found: {solution is not None}")
        print(f"  Time: {solve_time:.2f} seconds")
    
    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("results/data/parallel_comparison.csv", index=False)
    
    return results

def run_scaling_experiment():
    """
    Test scaling with increasing base size
    """
    print("\n" + "="*80)
    print("EXPERIMENT 5: Scaling with Base Size")
    print("="*80)
    
    base_sizes = [10, 16, 20, 26, 32, 36]
    results = []
    
    for base in base_sizes:
        print(f"Testing base-{base}...")
        
        # Create problem with this base
        generator = ProblemGenerator(
            base=base,
            max_word_length=4,
            variable_count=8
        )
        problem = generator.generate_simple_addition()
        
        # Use AC3 and Z3 solvers
        csp_solver = AC3Solver(problem)
        sat_solver = create_sat_solver(problem, "z3")
        
        # Run CSP solver
        start_time = time.time()
        csp_solution = csp_solver.solve(timeout=180)
        csp_time = time.time() - start_time
        
        # Run SAT solver
        start_time = time.time()
        sat_solution = sat_solver.solve(timeout=180)
        sat_time = time.time() - start_time
        
        # Record results
        results.append({
            "base": base,
            "solver": "CSP (AC3)",
            "solution_found": csp_solution is not None,
            "solution_time": csp_time,
            "nodes_explored": csp_solver.stats.get("nodes_explored", "N/A") if hasattr(csp_solver, "stats") else "N/A"
        })
        
        results.append({
            "base": base,
            "solver": "SAT (Z3)",
            "solution_found": sat_solution is not None,
            "solution_time": sat_time,
            "variables_count": sat_solver.stats.get("variables_count", "N/A") if hasattr(sat_solver, "stats") else "N/A",
            "clauses_count": sat_solver.stats.get("clauses_count", "N/A") if hasattr(sat_solver, "stats") else "N/A"
        })
        
        print(f"  CSP: Solution found: {csp_solution is not None}, Time: {csp_time:.2f}s")
        print(f"  SAT: Solution found: {sat_solution is not None}, Time: {sat_time:.2f}s")
    
    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("results/data/scaling_with_base.csv", index=False)
    
    # Create chart
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    
    csp_df = df[df["solver"] == "CSP (AC3)"]
    sat_df = df[df["solver"] == "SAT (Z3)"]
    
    plt.plot(csp_df["base"], csp_df["solution_time"], marker='o', label="CSP (AC3)")
    plt.plot(sat_df["base"], sat_df["solution_time"], marker='s', label="SAT (Z3)")
    
    plt.xlabel("Base Size")
    plt.ylabel("Solution Time (seconds)")
    plt.title("Scaling Performance with Base Size")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig("results/charts/scaling_with_base.png")
    
    return results

if __name__ == "__main__":
    start_time = time.time()
    
    print("Starting BTP-2 Experiments")
    print("==========================")
    
    # Run all experiments
    comparative_results = run_comparative_experiment()
    # pt_results_var, pt_results_base = run_phase_transition_experiment()
    #optimization_results = run_optimization_experiment()
    #parallel_results = run_parallel_experiment()
    #scaling_results = run_scaling_experiment()
    
    total_time = time.time() - start_time
    print("\nAll experiments completed in {:.2f} minutes".format(total_time / 60))
    print("Results saved to 'results/' directory")