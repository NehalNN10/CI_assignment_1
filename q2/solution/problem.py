from abc import ABC, abstractmethod

class problem(ABC):
    """
    Abstract class for a problem
    """

    @abstractmethod
    def chromosome(self):
        """
        Generates a chromosome
        """
        pass

    @abstractmethod
    def fitness_function(self):
        """
        Calculates the fitness value of each chromosome in the population
        """
        pass

    @abstractmethod
    def crossover(self):
        """
        Crosses over two chromosomes
        """
        pass

    @abstractmethod
    def mutation(self):
        """
        Mutates a chromosome
        """
        pass
