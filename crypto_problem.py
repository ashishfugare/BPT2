import random
from enum import Enum
from typing import Dict, List, Set, Tuple, Optional

class OperationType(Enum):
    ADDITION = "+"
    SUBTRACTION = "-"
    MULTIPLICATION = "*"
    DIVISION = "/"

class CryptoArithmeticProblem:
    """
    Representation of a cryptarithmetic problem.
    
    Supports multiple operations and constraints.
    """
    
    def __init__(self, base: int = 10):
        self.base = base
        self.variables: Set[str] = set()
        self.constraints: List[Tuple] = []
        self.first_letter_constraints: Set[str] = set()
        
    def add_constraint(self, left_operands: List[str], operation: OperationType, 
                       right_result: str):
        """
        Add a constraint to the problem.
        
        Example:
        For SEND + MORE = MONEY, call:
        add_constraint(["SEND", "MORE"], OperationType.ADDITION, "MONEY")
        """
        self.constraints.append((left_operands, operation, right_result))
        
        # Extract all unique letters (variables)
        for word in left_operands + [right_result]:
            for letter in word:
                self.variables.add(letter)
                
            # First letter cannot be zero
            self.first_letter_constraints.add(word[0])
    
    def generate_domains(self) -> Dict[str, List[int]]:
        """Generate initial domains for all variables."""
        domains = {}
        for var in self.variables:
            if var in self.first_letter_constraints:
                domains[var] = list(range(1, self.base))
            else:
                domains[var] = list(range(0, self.base))
        return domains
    
    def evaluate(self, assignment: Dict[str, int], constraint_idx: int = None) -> bool:
        """
        Evaluate whether the assignment satisfies the constraints.
        If constraint_idx is provided, only that constraint is evaluated.
        """
        constraints_to_check = (
            [self.constraints[constraint_idx]] if constraint_idx is not None 
            else self.constraints
        )
        
        for left_operands, operation, right_result in constraints_to_check:
            # Check if all variables in this constraint are assigned
            all_words = left_operands + [right_result]
            all_letters = set()
            for word in all_words:
                all_letters.update(set(word))
                
            if not all(var in assignment for var in all_letters):
                # Not all variables are assigned yet
                return True
            
            # Calculate values of operands
            left_values = []
            for word in left_operands:
                value = 0
                for i, letter in enumerate(reversed(word)):
                    value += assignment[letter] * (self.base ** i)
                left_values.append(value)
            
            # Calculate result
            right_value = 0
            for i, letter in enumerate(reversed(right_result)):
                right_value += assignment[letter] * (self.base ** i)
            
            # Check if constraint is satisfied
            if operation == OperationType.ADDITION:
                if sum(left_values) != right_value:
                    return False
            elif operation == OperationType.SUBTRACTION:
                if left_values[0] - sum(left_values[1:]) != right_value:
                    return False
            elif operation == OperationType.MULTIPLICATION:
                result = 1
                for val in left_values:
                    result *= val
                if result != right_value:
                    return False
            elif operation == OperationType.DIVISION:
                # Division needs special handling to avoid division by zero
                try:
                    result = left_values[0]
                    for val in left_values[1:]:
                        if val == 0:
                            return False
                        result /= val
                    if result != right_value:
                        return False
                except ZeroDivisionError:
                    return False
        
        return True
    
    def __repr__(self) -> str:
        """String representation of the problem."""
        problem_str = f"Cryptarithmetic Problem (Base {self.base}):\n"
        for left_operands, operation, right_result in self.constraints:
            problem_str += " ".join(left_operands)
            problem_str += f" {operation.value} "
            problem_str += right_result + "\n"
        return problem_str


class ProblemGenerator:
    """
    Generator for cryptarithmetic problems of varying complexity.
    """
    
    def __init__(self, base: int = 10, max_word_length: int = 5, 
                 variable_count: int = 8):
        self.base = base
        self.max_word_length = max_word_length
        self.variable_count = variable_count
        
    def generate_random_word(self, length: int, letters: List[str]) -> str:
        """Generate a random word using available letters."""
        return ''.join(random.choice(letters) for _ in range(length))
    
    def generate_simple_addition(self) -> CryptoArithmeticProblem:
        """Generate a simple addition problem (A + B = C)."""
        problem = CryptoArithmeticProblem(base=self.base)
        
        # Generate random letters for variables
        letters = [chr(ord('A') + i) for i in range(self.variable_count)]
        
        # Generate random word lengths
        word1_len = random.randint(2, self.max_word_length)
        word2_len = random.randint(2, self.max_word_length)
        result_len = max(word1_len, word2_len) + 1
        
        # Generate random words
        word1 = self.generate_random_word(word1_len, letters)
        word2 = self.generate_random_word(word2_len, letters)
        result = self.generate_random_word(result_len, letters)
        
        # Add constraint
        problem.add_constraint([word1, word2], OperationType.ADDITION, result)
        
        return problem
    
    def generate_multi_operation(self, op_count: int = 2) -> CryptoArithmeticProblem:
        """Generate a problem with multiple operations."""
        problem = CryptoArithmeticProblem(base=self.base)
        
        # Generate random letters for variables
        letters = [chr(ord('A') + i) for i in range(self.variable_count)]
        
        for _ in range(op_count):
            # Generate random operation
            operations = list(OperationType)
            operation = random.choice(operations)
            
            # Generate random number of operands (2-3)
            operand_count = random.randint(2, 3)
            
            # Generate random word lengths
            operand_lengths = [random.randint(2, self.max_word_length) 
                              for _ in range(operand_count)]
            result_len = max(operand_lengths) + 1
            
            # Generate random words
            operands = [self.generate_random_word(length, letters) 
                        for length in operand_lengths]
            result = self.generate_random_word(result_len, letters)
            
            # Add constraint
            problem.add_constraint(operands, operation, result)
        
        return problem
    
    def generate_multi_constraint(self, constraint_count: int = 2) -> CryptoArithmeticProblem:
        """Generate a problem with multiple constraints that share variables."""
        problem = CryptoArithmeticProblem(base=self.base)
        
        # Generate random letters for variables (slightly fewer than variable_count)
        # to ensure variable sharing across constraints
        actual_var_count = min(self.variable_count, 6)  # Ensure reuse of variables
        letters = [chr(ord('A') + i) for i in range(actual_var_count)]
        
        for _ in range(constraint_count):
            # Generate random operation
            operations = list(OperationType)
            operation = random.choice(operations)
            
            # Generate random word lengths
            word1_len = random.randint(2, self.max_word_length)
            word2_len = random.randint(2, self.max_word_length)
            result_len = max(word1_len, word2_len) + 1
            
            # Generate random words
            word1 = self.generate_random_word(word1_len, letters)
            word2 = self.generate_random_word(word2_len, letters)
            result = self.generate_random_word(result_len, letters)
            
            # Add constraint
            problem.add_constraint([word1, word2], operation, result)
        
        return problem


# Example usage
if __name__ == "__main__":
    # Create a classic SEND + MORE = MONEY problem
    classic_problem = CryptoArithmeticProblem()
    classic_problem.add_constraint(["SEND", "MORE"], OperationType.ADDITION, "MONEY")
    print(classic_problem)
    
    # Generate a random problem
    generator = ProblemGenerator(base=10, max_word_length=4, variable_count=8)
    
    # Generate different types of problems
    simple_problem = generator.generate_simple_addition()
    print("Simple Addition Problem:")
    print(simple_problem)
    
    multi_op_problem = generator.generate_multi_operation(op_count=2)
    print("Multi-Operation Problem:")
    print(multi_op_problem)
    
    multi_constraint_problem = generator.generate_multi_constraint(constraint_count=2)
    print("Multi-Constraint Problem:")
    print(multi_constraint_problem)