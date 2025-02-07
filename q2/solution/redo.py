import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image, ImageDraw
# import colour
from colour.difference import delta_E_CIE1976
class polygon:
    def __init__(self, vertices, color):
        self.vertices = vertices
        self.color = color

class monaLisa():
    def __init__(self, imagePath, numVertices, populationSize, numPolygons=50):
        self.inputImage = Image.open(imagePath).convert('RGBA')
        self.targetImage = self.inputImage.resize((200, 200))
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
        # for i in range(self.numPolygons):
        #     print(chromosome["vertices"][i])
        #     draw.polygon(chromosome["vertices"][i], fill=chromosome["color"])
        for polygon in chromosome["polygons"]:
            draw.polygon(polygon.vertices, fill=polygon.color)
        # draw.polygon(chromosome["vertices"], fill=chromosome["color"])
        return img

    def refined_fitness(self, chromosome):
        # Render the new drawing (chromosome)
        rendered_image = self.render_drawing(chromosome)
        rendered_pixels = np.array(rendered_image)

        # Compute color error
        # Normalize both images to [0, 1] range to make delta E more effective
        rendered_pixels = rendered_pixels / 255.0
        source_pixels = self.source_pixels / 255.0

        # Using delta E for color comparison (you can also try other color difference methods)
        color_diff = np.mean(delta_E_CIE1976(source_pixels, rendered_pixels))

        # You can combine other metrics (like structure) with the color_diff if needed
        return color_diff

    def render_polygon(self, polygon, size):
        img = Image.new("RGBA", size)
        draw = ImageDraw.Draw(img)
        draw.polygon(polygon.vertices, fill=polygon.color)
        return np.array(img)

    def bounding_box_fitness(self, chromosome, target_image):
        total_shape_error = 0

        # Iterate over each polygon in the chromosome
        for polygon in chromosome["polygons"]:
            # Calculate the bounding box of the polygon
            min_x = min([p[0] for p in polygon.vertices])
            max_x = max([p[0] for p in polygon.vertices])
            min_y = min([p[1] for p in polygon.vertices])
            max_y = max([p[1] for p in polygon.vertices])

            # Sample pixels within the bounding box region from the target image
            target_pixels = target_image[min_y:max_y, min_x:max_x]
            print(target_image.size)
            polygon_pixels = self.render_polygon(polygon, target_image.size)[
                min_y:max_y, min_x:max_y
            ]

            # Compute fitness based on how well the polygon's bounding box matches the target
            total_shape_error += np.sum(np.abs(target_pixels - polygon_pixels))

        return total_shape_error
        
    def chromosome_fitness(self, chromosome):
        # Calculate color fitness
        color_error = self.refined_fitness(chromosome)

        # Calculate shape fitness (bounding box or edge-based)
        shape_error = self.bounding_box_fitness(chromosome, self.source_pixels)

        # Combine the errors, giving more weight to either component if necessary
        total_fitness = color_error + shape_error  # You can use weights here if desired
        return total_fitness

    # def chromosome_fitness(self, chromosome):
    #     # Render the new drawing
    #     rendered_image = self.render_drawing(chromosome)
    #     rendered_pixels = np.array(rendered_image)

    #     # Calculate the error
    #     # ? not good enough
    #     # error = np.sum((rendered_pixels - self.source_pixels) ** 2)
    #     error = np.mean(
    #         delta_E_CIE1976(
    #             self.source_pixels, rendered_pixels
    #         )
    #     )
    #     # print(error)
    #     # input("Check")
    #     return error

    def chromosome(self):
        return {
            "polygons": [
                polygon(
                    vertices=[
                        (random.randint(0, self.height), random.randint(0, self.width))
                        for _ in range(self.numVertices)
                    ],
                    color=(
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                    ),
                )
                for _ in range(self.numPolygons)
            ],
            "image": None,
            "fitness": 0,
        }

    def initialize_population(self):
        self.population = [self.chromosome() for _ in range(self.populationSize)]

    def set_fitness(self):
        for chromosome in self.population:
            chromosome["fitness"] = self.chromosome_fitness(chromosome)

    # def crossover(self, parent1, parent2):
    #     # Determine crossover point
    #     crossover_point = random.randint(0, self.numPolygons)
    #     child1_polygons = parent1["polygons"][:crossover_point] + parent2["polygons"][crossover_point:]
    #     child2_polygons = parent2["polygons"][:crossover_point] + parent1["polygons"][crossover_point:]
    #     child1 = {"polygons": child1_polygons, "fitness": 0}
    #     child1["fitness"] = self.chromosome_fitness(child1)
    #     child2 = {"polygons": child2_polygons, "fitness": 0}
    #     child2["fitness"] = self.chromosome_fitness(child2)
    #     return child1, child2

    # def crossover(self, parent1, parent2):
    #     crossover_point = random.randint(0, self.numPolygons)
    #     child1_polygons = []
    #     child2_polygons = []

    #     for i in range(self.numPolygons):
    #         if i < crossover_point:
    #             # Blend colors from both parents
    #             new_color = tuple(
    #                 int((c1 + c2) / 2) for c1, c2 in zip(parent1["polygons"][i].color, parent2["polygons"][i].color)
    #             )
    #             new_vertices = random.choice([parent1["polygons"][i].vertices, parent2["polygons"][i].vertices])
    #         else:
    #             new_color = parent2["polygons"][i].color
    #             new_vertices = parent1["polygons"][i].vertices

    #         child1_polygons.append(polygon(new_vertices, new_color))
    #         child2_polygons.append(polygon(new_vertices, new_color))

    #     return {"polygons": child1_polygons, "fitness": 0}, {"polygons": child2_polygons, "fitness": 0}

    def crossover(self, parent1, parent2):
        strategies = ["split", "blend"]
        strategy = random.choices(strategies, weights=[0.5, 0.5], k=1)[0]
        print(strategy)
        if strategy == "split":
            childImage = Image.new("RGBA", (self.width, self.height))
            splitType = random.choice(["horizontal", "vertical", "quadrant"])
            if splitType == "horizontal":
                splitPos = random.randint(0, self.height)
                # upperPart = parent1["image"].crop((0, 0, self.width, splitPos))
                upperPart = self.render_drawing(parent1).crop((0, 0, self.width, splitPos))
                # lowerPart = parent2["image"].crop((0, splitPos, self.width, self.height))
                lowerPart = self.render_drawing(parent2).crop((0, splitPos, self.width, self.height))
                childImage.paste(upperPart, (0, 0))
                childImage.paste(lowerPart, (0, splitPos))
            elif splitType == "vertical":
                splitPos = random.randint(0, self.width)
                leftPart = self.render_drawing(parent1).crop(
                    (0, 0, splitPos, self.height)
                )
                rightPart = self.render_drawing(parent2).crop(
                    (splitPos, 0, self.width, self.height)
                )
                childImage.paste(leftPart, (0, 0))
                childImage.paste(rightPart, (splitPos, 0))
            else:  # quadrant - split
                splitPosX = random.randint(0, self.width)
                splitPosY = random.randint(0, self.height)
                topLeft = self.render_drawing(parent1).crop((0, 0, splitPosX, splitPosY))
                topRight = self.render_drawing(parent2).crop((splitPosX, 0, self.width, splitPosY))
                bottomLeft = self.render_drawing(parent1).crop(
                    (0, splitPosY, splitPosX, self.height)
                )
                bottomRight = self.render_drawing(parent2).crop(
                    (splitPosX, splitPosY, self.width, self.height)
                )
                childImage.paste(topLeft, (0, 0))
                childImage.paste(topRight, (splitPosX, 0))
                childImage.paste(bottomLeft, (0, splitPosY))
                childImage.paste(bottomRight, (splitPosX, splitPosY))
            # child = self.createChromosome()
            child = self.chromosome()
            child["image"] = childImage
            child["array"] = np.array(childImage)
            # child["fitness"] = self.fitnessFunction(child["array"])
            child["fitness"] = self.chromosome_fitness(child)
            print(child["fitness"], max(parent1["fitness"], parent2["fitness"]))
            if child["fitness"] <= max(parent1["fitness"], parent2["fitness"]):
                return child
        else:  # color blend
            blend = random.random()
            childImage = Image.blend(self.render_drawing(parent1), self.render_drawing(parent2), blend)
            child = self.chromosome()
            child["image"] = childImage
            child["array"] = np.array(childImage)
            # child["fitness"] = self.fitnessFunction(child["array"])
            child["fitness"] = self.chromosome_fitness(child)
            print(child["fitness"], max(parent1["fitness"], parent2["fitness"]))
            if child["fitness"] <= min(parent1["fitness"], parent2["fitness"]):
                return child
        return None
        # return child
    # TODO: modify so that it works

    def mutate(self, chromosome, mutation_rate=0.3):
        # TODO
        for polygon in chromosome["polygons"]:
            if random.random() < mutation_rate:
                # Mutate color
                polygon.color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )

            if random.random() < mutation_rate:
                # Mutate a single vertex instead of all
                idx = random.randint(0, len(polygon.vertices) - 1)
                polygon.vertices[idx] = (
                    random.randint(0, self.height),
                    random.randint(0, self.width),
                )

    def select_survivors(self): # Depends on selection scheme
        # TODO
        self.set_fitness()
        return sorted(self.population, key=lambda x: x["fitness"])[:self.populationSize]

    def total_fitness(self):
        tf = 0
        for chromosome in self.population:
            tf += self.chromosome_fitness(chromosome)
        return tf

    # For one parent
    # def select_parents(self):
    #     p = random.random()
    #     cumulative = 0
    #     for chromosome in self.population:
    #         print(chromosome)
    #         cumulative += self.chromosome_fitness(chromosome) / self.total_fitness()
    #         if cumulative >= p:
    #             return chromosome

    def select_parents(self):
        tournament_size = 4
        tournament = random.sample(self.population, tournament_size)
        return min(tournament, key=lambda x: x["fitness"])  # Select best

    # def evolve(self, num_generations):
    #     # TODO
    #     self.initialize_population()
    #     for generation in range(num_generations):
    #         next_generation = []
    #         for _ in range(self.populationSize // 2):
    #             parent1= self.select_parents()
    #             print(parent1)
    #             input("Check")
    #             parent2 = self.select_parents()
    #             child1, child2 = self.crossover(parent1, parent2)
    #             self.mutate(child1)
    #             self.mutate(child2)
    #             next_generation.append(child1)
    #             next_generation.append(child2)
    #         self.population = self.select_survivors()
    #         best_individual = min(self.population, key=self.chromosome_fitness)
    #         print(f"Generation {generation}: Best Fitness: {self.chromosome_fitness(best_individual)}")
    #         self.record_best_individual(best_individual, generation)

    def evolve(self, num_generations):
        average_fitness = {}
        self.initialize_population()

        for generation in range(num_generations):
            self.set_fitness()

            next_generation = []
            elites = sorted(self.population, key=lambda x: x["fitness"])[:2]  # Keep best 2
            next_generation.extend(elites)

            for _ in range((self.populationSize - 2) // 2):
                parent1 = self.select_parents()
                parent2 = self.select_parents()
                # child1, child2 = self.crossover(parent1, parent2)
                child1 = self.crossover(parent1, parent2)
                while (child1 == None):
                    child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent1, parent2)
                while child2 == None:
                    child2 = self.crossover(parent1, parent2)

                # Reduce mutation rate over generations
                mutation_rate = max(0.1, 0.3 - (generation / num_generations) * 0.2)
                self.mutate(child1, mutation_rate)
                self.mutate(child2, mutation_rate)

                next_generation.append(child1)
                next_generation.append(child2)

            # self.population = next_generation
            self.population = self.select_survivors()

            avg = sum(chromosome["fitness"] for chromosome in self.population) / self.populationSize
            average_fitness[generation] = avg

            best_individual = min(self.population, key=lambda x: x["fitness"])
            print(f"Generation {generation}: Best Fitness: {best_individual['fitness']}")
            self.record_best_individual(best_individual, generation)

        return average_fitness

    def record_best_individual(self, best_individual, generation):
        image = self.render_drawing(best_individual)
        image.save(f'best_ones/best_individual_generation_{generation}.png')

ga = monaLisa(imagePath="ml.bmp", numPolygons=50, numVertices=3, populationSize=50)
s = ga.evolve(num_generations=2000)
print(s)
