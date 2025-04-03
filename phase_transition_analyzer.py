import time
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd

from crypto_problem import CryptoArithmeticProblem, ProblemGenerator, OperationType
from csp_solver import CSPSolver, AC3Solver
from sat_solver import create_sat_solver


class PhaseTransitionAnalyzer:
    """
    Analyze phase transitions in cryptoarithmetic problems.
    
    Phase transitions are regions where problems transition from
    being easy to solve to being hard to solve.
    """
    
    def __init__(self, output_dir: str = "phase_transition_results"):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_controlled_difficulty_problems(self, 
                                              num_problems: int = 20, 
                                              param_values: List[Any] = None,
                                              param_name: str = "variable_count",
                                              base: int = 10) -> List[CryptoArithmeticProblem]:
        """
        Generate problems with controlled difficulty parameters.
        
        Args:
            num_problems: Number of problems for each parameter value
            param_values: List of parameter values to test
            param_name: Name of the parameter to vary ('variable_count', 'base', etc.)
            base: Base for the number system (10, 16, etc.) - if not varied
            
        Returns:
            List of generated problems with metadata
        """
        if param_values is None:
            if param_name == "variable_count":
                param_values = list(range(4, 15))
            elif param_name == "base":
                param_values = [4, 6, 8, 10, 12, 16, 20, 26, 32, 36]
            else:
                raise ValueError(f"No default values for parameter: {param_name}")
        
        problems = []
        
        for param_value in param_values:
            # Set up generator based on parameter being varied
            if param_name == "variable_count":
                generator = ProblemGenerator(
                    base=base,
                    max_word_length=param_value // 2,  # Adjust word length based on variable count
                    variable_count=param_value
                )
            elif param_name == "base":
                generator = ProblemGenerator(
                    base=param_value,
                    max_word_length=5,
                    variable_count=8
                )
            else:
                raise ValueError(f"Unsupported parameter: {param_name}")
            
            # Generate problems
            for _ in range(num_problems):
                problem = generator.generate_simple_addition()
                
                # Add metadata
                problem.metadata = {
                    "param_name": param_name,
                    "param_value": param_value,
                    "base": problem.base,
                    "variable_count": len(problem.variables),
                    "constraints_count": len(problem.constraints),
                }
                
                problems.append(problem)
        
        return problems
    
    def analyze_phase_transition(self, 
                               problems: List[CryptoArithmeticProblem],
                               solver_names: List[str] = None,
                               timeout: int = 60) -> pd.DataFrame:
        """
        Analyze phase transition by solving problems with varying difficulty.
        
        Args:
            problems: List of problems with controlled difficulty
            solver_names: List of solvers to test
            timeout: Maximum time in seconds to spend on each problem
            
        Returns:
            DataFrame with results
        """
        if solver_names is None:
            solver_names = ["csp_ac3", "sat_z3"]
        
        # Create solver factories
        solver_factories = {
            "csp_basic": lambda p: CSPSolver(p),
            "csp_ac3": lambda p: AC3Solver(p),
            "sat_z3": lambda p: create_sat_solver(p, "z3"),
            "sat_minisat": lambda p: create_sat_solver(p, "minisat")
        }
        
        results = []
        
        for problem in problems:
            param_name = problem.metadata["param_name"]
            param_value = problem.metadata["param_value"]
            
            for solver_name in solver_names:
                if solver_name not in solver_factories:
                    raise ValueError(f"Unknown solver: {solver_name}")
                
                # Create solver
                solver_factory = solver_factories[solver_name]
                solver = solver_factory(problem)
                
                # Solve and time
                start_time = time.time()
                solution = solver.solve(timeout=timeout)
                solve_time = time.time() - start_time
                
                # Record results
                result = {
                    "solver": solver_name,
                    param_name: param_value,
                    "base": problem.base,
                    "variable_count": len(problem.variables),
                    "constraints_count": len(problem.constraints),
                    "solution_found": solution is not None,
                    "solution_time": solve_time,
                    "timed_out": solve_time >= timeout * 0.99,  # Allow for small margin
                }
                
                # Add solver-specific stats
                if hasattr(solver, "stats"):
                    for key, value in solver.stats.items():
                        result[key] = value
                
                results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        import time as time_module
        timestamp = time_module.strftime("%Y%m%d-%H%M%S")
        filename = f"phase_transition_{timestamp}.csv"
        df.to_csv(f"{self.output_dir}/{filename}", index=False)
        
        return df
    
    def plot_phase_transition(self, results: pd.DataFrame) -> None:
        """
        Plot phase transition results.
        
        Args:
            results: DataFrame with phase transition analysis results
        """
        # Get parameter name (what was varied)
        param_names = [col for col in results.columns if col in ["variable_count", "base"]]
        if not param_names:
            raise ValueError("Could not determine parameter name")
        
        param_name = param_names[0]
        
        # Plot solution time vs. parameter value
        plt.figure(figsize=(12, 8))
        
        for solver in results['solver'].unique():
            solver_data = results[results['solver'] == solver]
            
            # Group by parameter value and calculate mean and standard deviation
            grouped = solver_data.groupby(param_name).agg({
                'solution_time': ['mean', 'std'],
                'solution_found': 'mean'
            })
            
            # Extract data for plotting
            x = grouped.index
            y = grouped['solution_time']['mean']
            yerr = grouped['solution_time']['std']
            
            plt.errorbar(x, y, yerr=yerr, marker='o', linestyle='-', label=solver)
        
        plt.xlabel(param_name.replace('_', ' ').title())
        plt.ylabel('Mean Solution Time (s)')
        plt.title(f'Phase Transition: Solution Time vs. {param_name.replace("_", " ").title()}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/phase_transition_time.png")
        
        # Plot success rate vs. parameter value
        plt.figure(figsize=(12, 8))
        
        for solver in results['solver'].unique():
            solver_data = results[results['solver'] == solver]
            
            # Group by parameter value and calculate success rate
            grouped = solver_data.groupby(param_name)['solution_found'].mean()
            
            plt.plot(grouped.index, grouped.values, marker='o', linestyle='-', label=solver)
        
        plt.xlabel(param_name.replace('_', ' ').title())
        plt.ylabel('Success Rate')
        plt.title(f'Phase Transition: Success Rate vs. {param_name.replace("_", " ").title()}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/phase_transition_success.png")
        
        # If available, plot search effort (nodes explored, backtracks) vs. parameter value
        if 'nodes_explored' in results.columns:
            plt.figure(figsize=(12, 8))
            
            for solver in results['solver'].unique():
                solver_data = results[results['solver'] == solver]
                
                if 'nodes_explored' not in solver_data.columns:
                    continue
                
                # Group by parameter value and calculate mean
                grouped = solver_data.groupby(param_name)['nodes_explored'].mean()
                
                plt.plot(grouped.index, grouped.values, marker='o', linestyle='-', label=solver)
            
            plt.xlabel(param_name.replace('_', ' ').title())
            plt.ylabel('Nodes Explored (mean)')
            plt.title(f'Search Effort: Nodes Explored vs. {param_name.replace("_", " ").title()}')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/phase_transition_nodes.png")
        
        # Log-scale plots can help identify phase transitions
        plt.figure(figsize=(12, 8))
        
        for solver in results['solver'].unique():
            solver_data = results[results['solver'] == solver]
            
            # Group by parameter value and calculate mean
            grouped = solver_data.groupby(param_name)['solution_time'].mean()
            
            plt.semilogy(grouped.index, grouped.values, marker='o', linestyle='-', label=solver)
        
        plt.xlabel(param_name.replace('_', ' ').title())
        plt.ylabel('Mean Solution Time (s) - Log Scale')
        plt.title(f'Phase Transition: Solution Time vs. {param_name.replace("_", " ").title()} (Log Scale)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/phase_transition_time_log.png")
        
        # Print analysis message
        print(f"Phase transition analysis complete. Results saved to {self.output_dir}")


def run_phase_transition_analysis():
    """
    Run a comprehensive phase transition analysis.
    """
    analyzer = PhaseTransitionAnalyzer(output_dir="phase_transition_results")
    
    # Analyze phase transition with variable count
    print("Analyzing phase transition with variable count...")
    
    problems_variable_count = analyzer.generate_controlled_difficulty_problems(
        num_problems=10,
        param_name="variable_count",
        param_values=list(range(4, 13)),
        base=10
    )
    
    results_variable_count = analyzer.analyze_phase_transition(
        problems=problems_variable_count,
        solver_names=["csp_ac3", "sat_z3"],
        timeout=120
    )
    
    analyzer.plot_phase_transition(results_variable_count)
    
    # Analyze phase transition with base
    print("Analyzing phase transition with base size...")
    
    problems_base = analyzer.generate_controlled_difficulty_problems(
        num_problems=10,
        param_name="base",
        param_values=[4, 6, 8, 10, 12, 16, 20, 26, 32],
        base=None  # Will be set by param_values
    )
    
    results_base = analyzer.analyze_phase_transition(
        problems=problems_base,
        solver_names=["csp_ac3", "sat_z3"],
        timeout=120
    )
    
    analyzer.plot_phase_transition(results_base)
    
    return results_variable_count, results_base


if __name__ == "__main__":
    # Run the phase transition analysis
    results_variable_count, results_base = run_phase_transition_analysis()
    
    # Print summary
    print("\nPhase Transition Analysis Summary:")
    print("Variable Count Analysis:")
    print(f"Total problems: {len(results_variable_count) // 2}")  # Divide by number of solvers
    print(f"Success rate: {results_variable_count['solution_found'].mean()*100:.1f}%")
    
    print("\nBase Size Analysis:")
    print(f"Total problems: {len(results_base) // 2}")  # Divide by number of solvers
    print(f"Success rate: {results_base['solution_found'].mean()*100:.1f}%")