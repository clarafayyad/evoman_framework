import numpy as np

# Applies crossover on both parents to create children.
def crossover(parent1, parent2):
        """
        Apply non-blending arithmetic crossover to create two children from two parents.

        Args:
        - parent1: A numpy array of the first parent's parameters.
        - parent2: A numpy array of the second parent's parameters.

        Returns:
        - child1, child2: Two numpy arrays representing the offspring.
        """

        # Initialize children
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)

        # Generate crossover
        for i in range(len(parent1)):
            alpha = np.random.rand()  # Random weight between 0 and 1
            child1[i] = alpha * parent1[i] + (1 - alpha) * parent2[i]
            child2[i] = (1 - alpha) * parent1[i] + alpha * parent2[i]

        return child1, child2

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
