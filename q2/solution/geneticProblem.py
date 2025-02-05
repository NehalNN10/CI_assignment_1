class geneticProblem:
    def __init__(self, populationSize):
        self.populationSize = populationSize

    def initializePopulation(self):
        raise NotImplementedError("Subclasses should implement this!")

    def fitnessFunction(self, chromosomes):
        raise NotImplementedError("Subclasses should implement this!")

    def crossover(self, parent1, parent2):
        raise NotImplementedError("Subclasses should implement this!")

    def mutate(self, chromosome, mutationRate):
        raise NotImplementedError("Subclasses should implement this!")
