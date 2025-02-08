import numpy as np
from PIL import Image, ImageDraw, ImagePath, ImageOps
import random
from chromosome import Chromosome # use classes defined in chromosome.py
from pandas import DataFrame as df # storing details in a dataframe and converting into csv
import math
import matplotlib.pyplot as plt # *

MIN_POINTS = 3
MAX_POINTS = 10
POLYGONS = 50

class gen_alg:
    def __init__(self, filename):
        self.target = Image.open(filename).convert('RGBA')

        self.length, self.width = self.target.size
        self.target_array = np.array(self.target) # Using numpy arrays allows for quicker processing

    def evolve(self, population_size, generations):
        data = {'generation': [], 'fitness': [], 'cross_over': [], 'pop_gen_used': [], 'image_size': []}

        population = []

        # Initialization
        for i in range(population_size):
            ch = Chromosome(self.length, self.width)
            ch.get_color_fitness(self.target)
            population.append(ch)

        # Evolution
        for i in range(generations):
            new_population = []
            fitness = float('inf')

            while (len(new_population) < population_size):
                p1 = self.tournament_select(population)
                p2 = self.tournament_select(population)

                # We want to store the fittest individual's fitness from every generation (record keeping)
                fitness = min(p1.fitness, p2.fitness, fitness)

                # We have different crossover methods, and we choose one randomly

                # These are the following crossover methods
                # 1. Blend crossover
                # 2. Single point crossover
                # 3. Multi point crossover

                # 30% of the time, we use blend crossover
                # 60% of the time, we use single point crossover
                # 10% of the time, we use pixel crossover

                # While Charmot's article talks of a different combination, this one works
                # well for our case
                # https://medium.com/@sebastian.charmot/genetic-algorithm-for-image-recreation-4ca546454aaa#:~:text=Combined%20Approach

                rand = random.uniform(0, 1)
                if rand < 0.3:
                    ch1 = self.cross_over(p1, p2)

                    while ch1 is None:
                        p1 = self.select(population)
                        p2 = self.select(population)
                        ch1 = self.cross_over(p1, p2)

                elif rand <= 0.9:
                    ch1 = self.cross_over_2(p1, p2, 0.5)

                    while ch1 is None:
                        p1 = self.select(population)
                        p2 = self.select(population)
                        ch1 = self.cross_over_2(p1, p2, 0.5)

                else:
                    ch1 = self.mutate(p1)

                    while ch1 is None:
                        p1 = self.select(population)
                        ch1 = self.mutate(p1)

                new_population.append(ch1)

            population = new_population

            # store records every 100 generations
            # running for 15000 generations, it would be too much overhead
            # to store every generation's data
            if i % 100 == 0 or i == generations - 1:

                # store all these records
                data['generation'].append(i)
                data['fitness'].append(fitness)
                data['average_fitness'].append(sum([x.fitness for x in population]) / len(population))
                data['cross_over'].append("crossover_1")
                data['pop_gen_used'].append("random_image_array_1")
                data['image_size'].append((self.length, self.width))

                print(f"Generation: {i}, Best fitness value: {fitness}")

                population.sort(key=lambda x: x.fitness)
                fittest = population[0]

                # Save the images
                fittest.image.save(f"best_ones/fittest_{i}.png")

                data_df = df(data)
                # exporting to csv inside loop to store results in case of crash
                data_df.to_csv("data.csv")

        data_df = df(data)
        data_df.to_csv("data.csv")

        population.sort(key=lambda x: x.fitness)
        fittest = population[0]

        return fittest

    # tournament selection worked best for our case
    # 4 size worked best for us
    def tournament_select(self, population, tournament_size=4):
        indices = np.random.choice(len(population), tournament_size)

        random_subset = [population[i] for i in indices]

        winner = None

        for i in random_subset:
            if winner is None or i.fitness < winner.fitness:
                winner = i

        return winner

    # def roulette_wheel_select(self, population):
    #     total_fitness = sum([x.fitness for x in population])

    #     rand = random.uniform(0, total_fitness)

    #     current = 0

    #     for i in population:
    #         current += i.fitness

    #         if current >= rand:
    #             return i

    def select(self, population):
        return self.tournament_select(population)

    # Blend crossover
    # We mix the alpha values of the two parents based on the value of a random number
    # Sometimes, there's only a different shade
    def cross_over(self, p1, p2):
        child = Chromosome(self.length, self.width)

        blend_alpha = random.random()

        # The higher the alpha value, the more the child will resemble the first parent
        child.image = Image.blend(p1.image, p2.image, blend_alpha)
        child.array = np.array(child.image)
        child.get_color_fitness(self.target)

        # Makes sure child is fitter than the parents
        if child.fitness == min(p1.fitness, p2.fitness, child.fitness):
            return child

        return None

    # Single point crossover
    # Split an image either horizontally or vertically, and combine the two halves
    # with the halves of the other parent
    def cross_over_2(self, p1, p2, horizontal_prob):
        rand = random.random() # to decide whether to split horizontally or vertically

        if rand <= horizontal_prob:
            split_point = random.randint(1, self.width)

            # take top half of p1
            first = np.ones((split_point, self.length))
            # used for combining the two halves - much easier than manually computing via loop
            first = np.vstack((first, np.zeros((self.width - split_point, self.length))))

        else:
            split_point = random.randint(1, self.length)

            first = np.ones((self.width, split_point))
            first = np.hstack((first, np.zeros((self.width, self.length - split_point))))

        second = 1 - first

        # convert 0s and 1s into 4 channel array (RGBA)
        first = np.dstack([first, first, first, first])
        second = np.dstack([second, second, second, second])

        # 1s in first and 0s in second will give the first parent's half
        fhalf = np.multiply(p1.array, first)
        shalf = np.multiply(p2.array, second)

        # combine the two halves
        child_array = np.add(fhalf, shalf)

        child = Chromosome(self.length, self.width)

        child.image = Image.fromarray(child_array.astype(np.uint8))
        child.array = child_array.astype(np.uint8)

        child.get_color_fitness(self.target)

        if child.fitness == min(p1.fitness, p2.fitness, child.fitness):
            return child

        return None

    # Pixel crossover
    # Creates a random mask
    # Selects pixels from either parent based on the mask
    # This process can get chaotic and not very aesthetic, so it is given a probability of 5%
    def cross_over_3(self, p1, p2):
        first = np.random.randint(2, size=(self.width, self.length, 4))

        second = 1 - first

        fhalf = np.multiply(p1.array, first)
        shalf = np.multiply(p2.array, second)

        child_array = np.add(fhalf, shalf)
        child = Chromosome(self.length, self.width)

        child.image = Image.fromarray(child_array.astype(np.uint8))
        child.array = child_array.astype(np.uint8)

        child.get_color_fitness(self.target)
        return child

    # Polygon mutation
    # Adds random polygons to the image depending on region
    def mutate(self, p1):
        rounds = random.randint(1, 3) # number of mutations - doing more than 3 can be chaotic
        region = random.randint(1, (self.length + self.width) // 4)

        img = p1.image

        for i in range(rounds):
            no_vertices = random.randint(MIN_POINTS, MAX_POINTS)
            region_x = random.randint(0, self.length)
            region_y = random.randint(0, self.width)

            xy = []
            for j in range(no_vertices):
                x = random.randint(region_x - region, region_x + region)
                y = random.randint(region_y - region, region_y + region)
                xy.append((x, y))

            img1 = ImageDraw.Draw(img)
            img1.polygon(xy, fill=p1.random_color())

        child = Chromosome(self.length, self.width)
        child.image = img
        child.array = child.to_array(img)
        child.get_color_fitness(self.target)

        return child

    # Pixel level mutation
    def mutate_2(self, p1):
        num_pix = 50 # change the values of 50 random pixels - too many can be chaotic

        for i in range(num_pix):
            x = random.randint(0, self.length-1)
            y = random.randint(0, self.width-1)
            z = random.randint(0, 3)

            p1.array[x][y][z] = p1.array[x][y][z] + random.randint(-10, 10)

        p1.image = self.to_image(p1.array)
        p1.get_color_fitness(self.target)

    def to_image(self, array):
        im = Image.fromarray(array)
        return im

    def to_array(self, img):
        return np.array(img)

def main():
    ga = gen_alg("ml.bmp")
    fittest = ga.evolve(100, 15000)
    plt.imshow(fittest.image)
    plt.show()
    # fittest.image.show()

if __name__ == "__main__":
    main()
