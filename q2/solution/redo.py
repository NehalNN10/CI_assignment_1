import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

class monaLisa():
    def __init__(self, imagePath, numPolygons=50, vertices, populationSize):
        self.inputImage = Image.open(imagePath).convert('RGBA')
        self.targetImage = inputImage.resize((200, 200))
        self.height, self.width = self.targetImage.size
        self.population = []
        self.numPolygons = numPolygons
        self.numVertices = numVertices
        self.populationSize = populationSize

        self.source_pixels = np.array(self.inputImage)

    # TODO: Implement using earlier code
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

    def render_drawing(self, chromosome):
        img = Image.new("RGBA", (self.height, self.width))
        draw = ImageDraw.Draw(img)
        for i in range(self.numPolygons):
            draw.polygon(chromosome["vertices"][i], fill=chromosome["color"])
        return img

    def chromosome_fitness(self, chromosome):
        # Render the new drawing
        rendered_image = self.render_drawing(chromosome)
        rendered_pixels = np.array(rendered_image)

        # Calculate the error
        error = np.sum((rendered_pixels - self.source_pixels) ** 2)
        return error

    def chromosome(self):
        return {
            "vertices": [
                (random.randint(0, self.height), random.randint(0, self.width))
                for _ in range(self.numVertices)
            ],
            "color": (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            ),
        }

    def initialize_population(self):
        self.population = [self.chromosome() for _ in range(self.populationSize)]
    
    def crossover(self, parent1, parent2):
        # Determine crossover point
        crossover_point = random.randint(0, self.numPolygons)
        child1_vertices = parent1["vertices"][:crossover_point] + parent2["vertices"][crossover_point:]
        child2_vertices = parent2["vertices"][:crossover_point] + parent1["vertices"][crossover_point:]
        child1 = {"vertices": child1_vertices, "color": parent1["color"]}
        child2 = {"vertices": child2_vertices, "color": parent2["color"]}
        return child1, child2

    def mutate(self, chromosome, mutation_rate):
        # TODO
        if random.random() < mutation_rate:
        chromosome["color"] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        if random.random() < mutation_rate:
            chromosome["vertices"] = [
                (random.randint(0, self.height), random.randint(0, self.width))
                for _ in range(self.numVertices)
            ]

    def select_survivors(self): # Depends on selection scheme
        # TODO
        pass

    def total_fitness(self):
        tf = 0
        for chromosome in self.population:
            tf += self.chromosome_fitness(chromosome)
        return tf

    # For one parent
    def select_parents(self):
        p = random.random()
        cumulative = 0
        for chromosome in self.population:
            cumulative += self.chromosome_fitness(chromosome) / self.total_fitness()
            if cumulative >= p:
                return chromosome


    def evolve(self, num_generations):
        # TODO
        for generation in range(num_generations):
            next_generation = []
            for _ in range(self.populationSize // 2):
                parent1, parent2 = self.select_parents()
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1)
                self.mutate(child2)
                next_generation.append(child1)
                next_generation.append(child2)
            self.population = self.select_survivors(next_generation)
            best_individual = min(self.population, key=self.chromosome_fitness)
            print(f"Generation {generation}: Best Fitness: {self.chromosome_fitness(best_individual)}")
            self.record_best_individual(best_individual, generation)

    def record_best_individual(self):
        image = self.render_drawing(best_individual)
        image.save(f'best_individual_generation_{generation}.png')