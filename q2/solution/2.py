from geneticProblem import geneticProblem
import matplotlib.pyplot as plt
# import colour
from colour.difference import delta_E_CIE1976
import copy
import numpy as np
from PIL import Image, ImageDraw
import random


class monaLisa(geneticProblem):
    def __init__(self, populationSize, filename, numPolygons=50, numVertices=3):
        inputImage = Image.open(filename)
        self.targetImage = inputImage.resize((200, 200))
        self.height, self.width = self.targetImage.size
        self.population = []
        self.numPolygons = numPolygons
        self.numVertices = numVertices
        self.populationSize = populationSize

    # This function is used to generate individual images. Random num of polygons are generated. We have added
    # the functionality to load the original target image to get the colors of the pixel at t a particular point. Which
    # we have then used with variations for the polygon color. This approach has given us great results.
    def genImage(self, height, width):
        numPolygons = random.randint(3, min(self.numPolygons, 6))
        img = Image.new("RGBA", (height, width))
        pix = self.targetImage.load()
        draw = ImageDraw.Draw(img)
        for _ in range(numPolygons):
            numVertices = self.numVertices
            polygon = []
            x_center = random.randint(0, height)
            y_center = random.randint(0, width)
            for _ in range(numVertices):
                x = random.randint(x_center - 10, x_center + 10)
                y = random.randint(y_center - 10, y_center + 10)
                polygon.append((x, y))
            base_color = pix[x_center % width, y_center % height]
            color_variation = tuple(
                random.randint(max(colr - 10, 0), min(colr + 10, 255))
                for colr in base_color
            )
            draw.polygon(polygon, fill=color_variation)
        return img

    # mean of Delta E has been used to calculate the fitness, lesser the value, greater the fitness
    def fitnessFunction(self, chromosomes):
        return np.mean(
            # colour.difference.delta_e.delta_E_CIE1976(self.targetImage, chromosomes)
            delta_E_CIE1976(self.targetImage, chromosomes)
        )

    # Basically a potentiol solution - describes in detail in the report
    def createChromosome(self):
        chromosome = {}
        chromosome["height"] = self.height
        chromosome["width"] = self.width
        chromosome["fitness"] = float("inf")
        chromosome["image"] = self.genImage(self.height, self.width)
        chromosome["array"] = np.array(chromosome["image"])
        chromosome["fitness"] = self.fitnessFunction(chromosome["array"])
        return chromosome

    def initializePopulation(self):
        self.population = [self.createChromosome() for _ in range(self.populationSize)]

    # For crossover we have used a combination of two techniques. First is the color blend approach. Second is the split
    # approach. Using a combination by giving equal probabs to both approaches has given us great results. In the split
    # approach we have further used 3 types of splits - vertical, horizontal and quadrant split. The approach of
    # using quadrant split improved our results alot. What it basically does is thet  it splits both horizontally
    # and vertically, making 4 quadrants, and takes top left from parent1, top right from parent2, bottom left from parent1,
    # bottom right from parent2, and combines them.
    def crossover(self, parent1, parent2):
        strategies = ["split", "blend"]
        strategy = random.choices(strategies, weights=[0.5, 0.5], k=1)[0]
        if strategy == "split":
            childImage = Image.new("RGBA", (self.width, self.height))
            splitType = random.choice(["horizontal", "vertical", "quadrant"])
            if splitType == "horizontal":
                splitPos = random.randint(0, self.height)
                upperPart = parent1["image"].crop((0, 0, self.width, splitPos))
                lowerPart = parent2["image"].crop(
                    (0, splitPos, self.width, self.height)
                )
                childImage.paste(upperPart, (0, 0))
                childImage.paste(lowerPart, (0, splitPos))
            elif splitType == "vertical":
                splitPos = random.randint(0, self.width)
                leftPart = parent1["image"].crop((0, 0, splitPos, self.height))
                rightPart = parent2["image"].crop(
                    (splitPos, 0, self.width, self.height)
                )
                childImage.paste(leftPart, (0, 0))
                childImage.paste(rightPart, (splitPos, 0))
            else:  # quadrant - split
                splitPosX = random.randint(0, self.width)
                splitPosY = random.randint(0, self.height)
                topLeft = parent1["image"].crop((0, 0, splitPosX, splitPosY))
                topRight = parent2["image"].crop((splitPosX, 0, self.width, splitPosY))
                bottomLeft = parent1["image"].crop(
                    (0, splitPosY, splitPosX, self.height)
                )
                bottomRight = parent2["image"].crop(
                    (splitPosX, splitPosY, self.width, self.height)
                )
                childImage.paste(topLeft, (0, 0))
                childImage.paste(topRight, (splitPosX, 0))
                childImage.paste(bottomLeft, (0, splitPosY))
                childImage.paste(bottomRight, (splitPosX, splitPosY))
            child = self.createChromosome()
            child["image"] = childImage
            child["array"] = np.array(childImage)
            child["fitness"] = self.fitnessFunction(child["array"])
            if child["fitness"] <= max(parent1["fitness"], parent2["fitness"]):
                return child
        else:  # color blend
            blend = random.random()
            childImage = Image.blend(parent1["image"], parent2["image"], blend)
            child = self.createChromosome()
            child["image"] = childImage
            child["array"] = np.array(childImage)
            child["fitness"] = self.fitnessFunction(child["array"])
            if child["fitness"] <= min(parent1["fitness"], parent2["fitness"]):
                return child
        return None

    # This function mutates the image in a chromosome by randomly adding colored polygon shapes on top of the image.
    # To get better results, we have added a fitness scale to ensure that better chromosomes mutate less
    def mutate(self, chromosome, mutationRate=0.9):
        fitness_scale = 1 / (chromosome["fitness"] + 1e-3)
        if random.random() < mutationRate:
            area = random.randint(
                1, max(chromosome["height"], chromosome["width"]) // 10
            )
            img = chromosome["image"].copy()
            imgGen = ImageDraw.Draw(img)
            # The number of mutations is scaled by fitness; better fitness = fewer mutations
            maxMutations = max(1, int(fitness_scale * 5))
            numMutations = random.randint(1, maxMutations)
            for _ in range(numMutations):
                x = random.randint(area, chromosome["height"] - area)
                y = random.randint(area, chromosome["width"] - area)
                loc = [
                    (
                        random.randint(x - area, x + area),
                        random.randint(y - area, y + area),
                    )
                    for _ in range(self.numVertices)
                ]
                color = "#" + "".join(random.choices("0123456789ABCDEF", k=6))
                imgGen.polygon(loc, fill=color)
            mutatedChromosome = self.createChromosome()
            mutatedChromosome["image"] = img
            mutatedChromosome["array"] = np.array(img)
            mutatedChromosome["fitness"] = self.fitnessFunction(
                mutatedChromosome["array"]
            )
            return mutatedChromosome
        else:
            return chromosome


def rankSelect(population):
    population.sort(key=lambda ind: ind["fitness"])
    return random.choice(population[: len(population) // 2])


def tournamentSelect(population):
    indices = np.random.choice(len(population), 4)
    tournament = [population[i] for i in indices]
    return min(tournament, key=lambda indiv: indiv["fitness"])


# This works like a normal EA, Elitism has been applied to yield better results
def evolutionaryAlgorithm(question, generations, parentScheme):
    question.initializePopulation()
    numElites = 2

    def getBest(population):
        return min(ind["fitness"] for ind in population)

    for i in range(generations):
        question.population.sort(key=lambda ind: ind["fitness"])
        bestFit = getBest(question.population)
        elites = copy.deepcopy(question.population[:numElites])
        newPopulation = elites
        while len(newPopulation) < question.populationSize:
            parentOne = parentScheme(question.population)
            parentTwo = parentScheme(question.population)
            bestFit = min(parentOne["fitness"], parentTwo["fitness"], bestFit)
            child = question.crossover(parentOne, parentTwo)
            while child is None:
                parentOne = parentScheme(question.population)
                parentTwo = parentScheme(question.population)
                child = question.crossover(parentOne, parentTwo)
            child = question.mutate(child)
            newPopulation.append(child)
        newPopulation[-len(elites) :] = elites
        question.population = newPopulation
        print(
            "Fittest Chromosome in generation {} has a fitness of: {:.2f}".format(
                i, bestFit
            )
        )
        if i % 100 == 0 or i == generations - 1:
            question.population.sort(key=lambda ind: ind["fitness"])
            fittest = question.population[0]
            fittest["image"].save("MonaLisaResults/AMona-fittest_{}.png".format(i))
    question.population.sort(key=lambda ind: ind["fitness"])
    return question.population[0]


def main():
    path = "ml.bmp"
    populationSize = 100
    numPolygons = 50
    numVertices = 3
    numGenerations = 10000
    question = monaLisa(populationSize, path, numPolygons, numVertices)
    fittest_chromosome = evolutionaryAlgorithm(
        question, numGenerations, tournamentSelect
    )
    plt.imshow(fittest_chromosome["image"])
    plt.show()


if __name__ == "__main__":
    main()
