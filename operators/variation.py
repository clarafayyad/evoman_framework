import numpy as np

# Applies crossover on both parents to create children.
def crossover(parent1, parent2):
    pass

def mutation(child, mutation_rate=0.1, sigma=0.1):
        """
        Apply Gaussian mutation to neural network parameters.

        Args:
        - child: A numpy array of the neural network parameters.
        - mutation_rate: The probability of mutating each parameter.
        - sigma: Standard deviation of the Gaussian distribution for mutation.

        Returns:
        - mutated_child: A numpy array with mutated parameters.
        """

        mutated_child = np.copy(child)

        for i in range(len(mutated_child)):
            if np.random.rand() < mutation_rate:
                mutated_child[i] += mutated_child[i] + np.random.normal(0, sigma)

        return mutated_child
