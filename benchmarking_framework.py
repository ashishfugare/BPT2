import time
import csv
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any, Callable
import pandas as pd

from crypto_problem import CryptoArithmeticProblem, ProblemGenerator, OperationType
from csp_solver import (
    CSPSolver, MRVSolver, ForwardCheckingSolver, AC3Solver, 
    ConflictDirectedBackjumpingSolver, NogoodLearningCSPSolver, RandomRestartCSPSolver
)
from sat_solver import create_sat_solver


class BenchmarkRunner:
    """
    Framework for running benchmarks on different solvers.
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Available solvers
        self.solver_factories = {
            # CSP Solvers
            "csp_basic": lambda p: CSPSolver(p),
            "csp_mrv": lambda p: MRVSolver(p),
            "csp_forward_checking": lambda p: ForwardCheckingSolver(p),
            "csp_ac3": lambda p: AC3Solver(p),
            "csp_backjumping": lambda p: ConflictDirectedBackjumpingSolver(p),
            "csp_nogood": lambda p: NogoodLearningCSPSolver(p),
            "csp_random_restart": lambda p: RandomRestartCSPSolver(p, num_restarts=3),
            
            # SAT Solvers
            "sat_z3": lambda p: create_sat_solver(p, "z3"),
            "sat_minisat": lambda p: create_sat_solver(p, "minisat")
        }
    
    def generate_benchmark_problems(self, 
                                   num_problems: int = 10, 
                                   base: int = 10,
                                   max_word_length: int = 5,
                                   variable_count: int = 8,
                                   problem_types: List[str] = None) -> List[CryptoArithmeticProblem]:
        """
        Generate a set of benchmark problems.
        
        Args:
            num_problems: Number of problems to generate
            base: Base for the number system (10, 16, etc.)
            max_word_length: Maximum length of generated words
            variable_count: Target number of variables
            problem_types: List of problem types to generate ('simple', 'multi_op', 'multi_constraint')
                           If None, all types are generated
        
        Returns:
            List of generated problems
        """
        if problem_types is None:
            problem_types = ['simple', 'multi_op', 'multi_constraint']
        
        generator = ProblemGenerator(
            base=base,
            max_word_length=max_word_length,
            variable_count=variable_count
        )
        
        problems = []
        
        # Generate problems of each type
        for _ in range(num_problems):
            for problem_type in problem_types:
                if problem_type == 'simple':
                    problem = generator.generate_simple_addition()
                    problem.metadata = {
                        "type": "simple",
                        "base": base,
                        "variable_count": len(problem.variables),
                        "constraints_count": len(problem.constraints),
                    }
                    problems.append(problem)
                
                elif problem_type == 'multi_op':
                    problem = generator.generate_multi_operation(op_count=2)
                    problem.metadata = {
                        "type": "multi_op",
                        "base": base,
                        "variable_count": len(problem.variables),
                        "constraints_count": len(problem.constraints),
                    }
                    problems.append(problem)
                
                elif problem_type == 'multi_constraint':
                    problem = generator.generate_multi_constraint(constraint_count=2)
                    problem.metadata = {
                        "type": "multi_constraint",
                        "base": base,
                        "variable_count": len(problem.variables),
                        "constraints_count": len(problem.constraints),
                    }
                    problems.append(problem)
        
        return problems
    
    def run_solver(self, solver_name: str, problem: CryptoArithmeticProblem, 
                  timeout: int = 60) -> Dict[str, Any]:
        """
        Run a solver on a problem and return results.
        
        Args:
            solver_name: Name of the solver to use
            problem: Problem to solve
            timeout: Maximum time in seconds to spend solving
            
        Returns:
            Dictionary with benchmark results
        """
        if solver_name not in self.solver_factories:
            raise ValueError(f"Unknown solver: {solver_name}")
        
        # Create solver
        solver_factory = self.solver_factories[solver_name]
        solver = solver_factory(problem)
        
        # Track memory usage (if psutil is available)
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            memory_before = None
        
        # Solve and time
        start_time = time.time()
        solution = solver.solve(timeout=timeout)
        solve_time = time.time() - start_time
        
        # Get memory usage after solving
        try:
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = memory_after - memory_before
        except:
            memory_delta = None
        
        # Prepare results
        results = {
            "solver": solver_name,
            "problem_type": problem.metadata["type"] if hasattr(problem, "metadata") else "unknown",
            "base": problem.base,
            "variable_count": len(problem.variables),
            "constraints_count": len(problem.constraints),
            "solution_found": solution is not None,
            "solution_time": solve_time,
            "memory_delta_mb": memory_delta,
        }
        
        # Add solver-specific stats
        if hasattr(solver, "stats"):
            results.update(solver.stats)
        
        return results
    
    def run_benchmark(self, problems: List[CryptoArithmeticProblem], 
                     solver_names: List[str] = None,
                     timeout: int = 60,
                     parallel: bool = True,
                     max_workers: int = None) -> pd.DataFrame:
        """
        Run benchmark on multiple problems and solvers.
        
        Args:
            problems: List of problems to benchmark
            solver_names: List of solver names to use (if None, use all)
            timeout: Maximum time in seconds to spend on each problem
            parallel: Whether to run benchmarks in parallel
            max_workers: Maximum number of parallel workers (default: # of CPUs)
            
        Returns:
            DataFrame with benchmark results
        """
        if solver_names is None:
            solver_names = list(self.solver_factories.keys())
        
        # Create all benchmark tasks
        tasks = []
        for problem_idx, problem in enumerate(problems):
            for solver_name in solver_names:
                tasks.append((problem_idx, solver_name, problem))
        
        results = []
        
        if parallel and len(tasks) > 1:
            # Run tasks in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.run_solver, solver_name, problem, timeout): 
                    (problem_idx, solver_name) 
                    for problem_idx, solver_name, problem in tasks
                }
                
                for future in as_completed(futures):
                    problem_idx, solver_name = futures[future]
                    try:
                        result = future.result()
                        result["problem_idx"] = problem_idx
                        results.append(result)
                    except Exception as e:
                        print(f"Error running {solver_name} on problem {problem_idx}: {e}")
        else:
            # Run tasks sequentially
            for problem_idx, solver_name, problem in tasks:
                try:
                    result = self.run_solver(solver_name, problem, timeout)
                    result["problem_idx"] = problem_idx
                    results.append(result)
                except Exception as e:
                    print(f"Error running {solver_name} on problem {problem_idx}: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"benchmark_{timestamp}.csv"
        df.to_csv(os.path.join(self.output_dir, filename), index=False)
        
        return df
    
    def analyze_results(self, results: pd.DataFrame) -> None:
        """
        Analyze and visualize benchmark results.
        
        Args:
            results: DataFrame with benchmark results
        """
        # Create output directory for plots
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate summary statistics
        summary = results.groupby(['solver', 'problem_type']).agg({
            'solution_found': 'mean',
            'solution_time': ['mean', 'median', 'std', 'min', 'max'],
            'memory_delta_mb': ['mean', 'median', 'std', 'min', 'max'],
            'nodes_explored': ['mean', 'median', 'std', 'min', 'max'],
            'backtracks': ['mean', 'median', 'std', 'min', 'max'],
        }).reset_index()
        
        # Save summary
        summary.to_csv(os.path.join(self.output_dir, "summary_stats.csv"))
        
        # Plot solution times by solver and problem type
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(111)
        
        bar_width = 0.15
        opacity = 0.8
        problem_types = results['problem_type'].unique()
        solver_names = results['solver'].unique()
        
        for i, solver in enumerate(solver_names):
            solver_data = results[results['solver'] == solver]
            means = [solver_data[solver_data['problem_type'] == pt]['solution_time'].mean() 
                    for pt in problem_types]
            
            pos = np.arange(len(problem_types)) + bar_width * i
            plt.bar(pos, means, bar_width,
                    alpha=opacity,
                    label=solver)
        
        plt.xlabel('Problem Type')
        plt.ylabel('Average Solution Time (s)')
        plt.title('Average Solution Time by Solver and Problem Type')
        plt.xticks(np.arange(len(problem_types)) + bar_width * (len(solver_names) - 1) / 2, problem_types)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "solution_time_by_problem_type.png"))
        
        # Plot solution time vs. variable count
        plt.figure(figsize=(12, 8))
        
        for solver in solver_names:
            solver_data = results[results['solver'] == solver]
            plt.scatter(solver_data['variable_count'], solver_data['solution_time'],
                       label=solver, alpha=0.7)
        
        plt.xlabel('Number of Variables')
        plt.ylabel('Solution Time (s)')
        plt.title('Solution Time vs. Number of Variables')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "solution_time_vs_variables.png"))
        
        # Plot success rate by problem type
        plt.figure(figsize=(12, 8))
        
        success_rate = results.groupby(['solver', 'problem_type'])['solution_found'].mean().unstack()
        success_rate.plot(kind='bar', figsize=(12, 8))
        
        plt.ylabel('Success Rate')
        plt.title('Success Rate by Solver and Problem Type')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "success_rate.png"))
        
        # Plot scalability (solution time vs. base size)
        if len(results['base'].unique()) > 1:
            plt.figure(figsize=(12, 8))
            
            for solver in solver_names:
                solver_data = results[results['solver'] == solver]
                base_groups = solver_data.groupby('base')['solution_time'].mean()
                
                plt.plot(base_groups.index, base_groups.values, 'o-', label=solver)
            
            plt.xlabel('Base Size')
            plt.ylabel('Average Solution Time (s)')
            plt.title('Scalability: Average Solution Time vs. Base Size')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "scalability_base_size.png"))
        
        # Print summary message
        print(f"Analysis complete. Results saved to {self.output_dir}")
        print(f"Plots saved to {plots_dir}")


def run_comparative_benchmark():
    """
    Run a comprehensive benchmark comparing CSP and SAT approaches.
    """
    benchmark = BenchmarkRunner(output_dir="benchmark_results")
    
    # Generate problems with different sizes and complexities
    print("Generating benchmark problems...")
    
    # Base-10 problems of varying complexity
    problems_base10 = benchmark.generate_benchmark_problems(
        num_problems=5,
        base=10,
        max_word_length=5,
        variable_count=8,
        problem_types=['simple', 'multi_op', 'multi_constraint']
    )
    
    # Base-16 problems for scaling analysis
    problems_base16 = benchmark.generate_benchmark_problems(
        num_problems=5,
        base=16,
        max_word_length=5,
        variable_count=10,
        problem_types=['simple', 'multi_op']
    )
    
    # Base-32 problems for scaling analysis
    problems_base32 = benchmark.generate_benchmark_problems(
        num_problems=3,
        base=32,
        max_word_length=4,
        variable_count=12,
        problem_types=['simple']
    )
    
    # Combine all problems
    all_problems = problems_base10 + problems_base16 + problems_base32
    
    # Define solvers to benchmark
    solver_names = [
        # CSP Solvers - basic vs. optimized
        "csp_basic",
        "csp_mrv",
        "csp_forward_checking",
        "csp_ac3",
        "csp_backjumping",
        "csp_nogood",
        "csp_random_restart",
        
        # SAT Solvers
        "sat_z3",
        "sat_minisat"
    ]
    
    # Run the benchmark
    print(f"Running benchmark with {len(all_problems)} problems and {len(solver_names)} solvers...")
    results = benchmark.run_benchmark(
        problems=all_problems,
        solver_names=solver_names,
        timeout=300,  # 5 minutes per problem
        parallel=True
    )
    
    # Analyze results
    print("Analyzing results...")
    benchmark.analyze_results(results)
    
    return results


if __name__ == "__main__":
    # Run the full benchmark suite
    results = run_comparative_benchmark()
    
    # Print summary
    print("\nBenchmark Summary:")
    print(f"Total problems: {results['problem_idx'].nunique()}")
    print(f"Total solvers: {results['solver'].nunique()}")
    print(f"Total benchmark runs: {len(results)}")
    print(f"Successful solutions: {results['solution_found'].sum()} ({results['solution_found'].mean()*100:.1f}%)")
    print(f"Average solution time: {results['solution_time'].mean():.3f}s")