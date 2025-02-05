import random
import os
from PIL import Image, ImageDraw
import numpy as np

minPoints = 3
active_polygons_min = 1
canvas_width = 200
canvas_height = 200
mutation_rate = 0.3
max_polygons = 50
max_drawings = 50

# Class for polygons
class DnaPolygon:
    points = []
    brush = None

    def __init__(self, points, brush):
        self.points = points
        self.brush = brush

    def get_random(self, width, height):
        # TODO
        pts = []

        origin = DnaPoint(0, 0).get_random(width, height)
        for i in range(minPoints):
            pt = DnaPoint(0, 0).get_random(width, height)
            pts.append(pt)

        brush = DnaBrush(0, 0, 0, 0).get_random()
        # print(brush)
        # input("Now?")

        return DnaPolygon(pts, brush)

    def get_points(self):
        arr = []
        for point in self.points:
            # print(self.points)
            # input("Now?")
            arr.append((point.x, point.y))
        return arr

    def clone(self):
        clonedPoints = points.copy()
        clonedBrush = brush.clone()
        return DnaPolygon(clonedPoints, clonedBrush)

class DnaPoint:
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def get_random(self, width, height):
        point = DnaPoint(0, 0)
        point.x = random.randint(0, width)
        point.y = random.randint(0, height)
        return point

    def clone(self):
        return DnaPoint(self.x, self.y)

class DnaBrush:
    r = 0
    g = 0
    b = 0
    a = 0

    def __init__(self, r, g, b, a):
        self.r = r
        self.g = g
        self.b = b
        self.a = a

    def get_random(self):
        br = DnaBrush(0, 0, 0, 0)
        br.r = random.randint(0, 255)
        br.g = random.randint(0, 255)
        br.b = random.randint(0, 255)
        br.a = random.randint(0, 255)
        return br

    def clone(self):
        return DnaBrush(self.r, self.g, self.b, self.a)

# DnaDrawing class
class DnaDrawing:

    # Attributes
    width = 0
    height = 0
    is_dirty = False
    polygons = []

    def __init__(self, width, height, polygons=max_polygons, isDirty=True):
        self.width = width
        self.height = height
        self.polygons = polygons
        self.is_dirty = isDirty

    def point_count(self):
        count = 0
        for poly in polygons:
            count += len(poly.points)
        return count

    def set_dirty(self):
        self.is_dirty = True

    def get_random(self, width, height):
        drawing = DnaDrawing(width, height, [])
        for i in range(max_polygons):
            drawing.polygons.append(DnaPolygon([], None).get_random(width=width, height=height))
        return drawing

    def clone(self):
        clonedPolygons = []
        for poly in polygons:
            clonedPolygons.append(poly.clone())
        return DnaDrawing(width, height, clonedPolygons, is_dirty)

    def mutate(self, mutation_rate=mutation_rate):
        for polygon in self.polygons:
            if random.uniform(0, 1) < mutation_rate:
                # Mutate the polygon
                colors = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
                polygon.brush.r = colors[0]
                polygon.brush.g = colors[1]
                polygon.brush.b = colors[2]
                polygon.brush.a = colors[3]
            if random.random() < mutation_rate:
                # Mutate position
                polygon.points = [
                    DnaPoint(random.randint(0, canvas_width), random.randint(0, canvas_height))
                    for _ in polygon.points
                ]


class Pixel:
    r = 0
    g = 0
    b = 0
    a = 0

    def __init__(self, r, g, b, a):
        self.r = r
        self.g = g
        self.b = b
        self.a = a

    def get_pixel(self, img, x, y):
        r, g, b, a = img.getpixel((x, y))
        return Pixel(r, g, b, a)

class FitnessCalculator:
    def __init__(self, source_bitmap):
        self.source_bitmap = Image.open(source_bitmap).convert('RGBA')
        self.source_pixels = np.array(self.source_bitmap)

    def get_drawing_fitness(self, new_drawing):
        # Render the new drawing
        rendered_image = self.render_drawing(new_drawing)
        rendered_pixels = np.array(rendered_image)

        # Calculate the error
        error = np.sum((rendered_pixels - self.source_pixels) ** 2)
        return error

    def render_drawing(self, new_drawing):
        # Create a blank image with the same size as the source image
        rendered_image = Image.new('RGBA', self.source_bitmap.size)
        draw = ImageDraw.Draw(rendered_image)

        # Render the new drawing onto the blank image
        for shape in new_drawing.polygons:
            draw.polygon(shape.get_points(), fill=(shape.brush.r, shape.brush.g, shape.brush.b, shape.brush.a))

        return rendered_image

# Depending on selection scheme, this may change
class Crossover:

    p1 = None
    p2 = None

    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def cross(self):
        # Determine random crossover point
        crossover_point = random.randint(0, min(len(self.p1.polygons), len(self.p2.polygons))-1)

        # Create children polygon lists
        child_one_polygons = self.p1.polygons[:crossover_point] + self.p2.polygons[crossover_point:]
        child_two_polygons = self.p2.polygons[:crossover_point] + self.p1.polygons[crossover_point:]

        child_one = DnaDrawing(self.p1.width, self.p1.height, child_one_polygons)
        child_two = DnaDrawing(self.p2.width, self.p2.height, child_two_polygons)

        return child_one, child_two

class Selection:

    def __init__(self, population):
        self.population = population

    def select_parent(self, fitness_calculator):
        # TODO
        # Change as per selection scheme

        # Roulette based drawing
        total_fitness = sum([fitness_calculator.get_drawing_fitness(individual) for individual in self.population])

        # Pick a random fitness value and select the corresponding individual
        selection_value = random.uniform(0, total_fitness)
        cumulative_fitness = 0
        for individual in self.population:
            cumulative_fitness += fitness_calculator.get_drawing_fitness(individual)
            if cumulative_fitness >= selection_value:
                return individual

class Mutation:
    @staticmethod
    def mutate(dna_drawing, mutation_rate=mutation_rate):
        for polygon in dna_drawing.polygons:
            if random.uniform(0, 1) < mutation_rate:
                # Mutate the polygon
                colors = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
                polygon.brush.r = colors[0]
                polygon.brush.g = colors[1]
                polygon.brush.b = colors[2]
                polygon.brush.a = colors[3]
            if random.random() < mutation_rate:
                # Mutate position
                polygon.points = [
                    (random.randint(0, canvas_width), random.randint(0, canvas_height))
                    for _ in polygon.points
                ]


class EvolutionEngine:
    source_bitmap = None
    population_size = 0
    fitness_calculator = None
    crossover = None
    selection = None
    generations = 0
    fitnesses = dict()

    def __init__(self, source_bitmap, population_size, generations):
        self.source_bitmap = source_bitmap
        self.population_size = population_size
        self.generations = generations
        self.fitness_calculator = FitnessCalculator(source_bitmap)

    def evolve(self, generations):
        # TODO
        # Initialize population
        population = self.initialize_population()
        # print(population)
        Id = 0

        # Calculate fitness for each individual
        for gen in range(self.generations):
            print(f"Generation {gen + 1}")

            # Evaluate fitness of the population
            for individual in population:
                self.fitnesses[Id] = self.fitness_calculator.get_drawing_fitness(individual)
                Id += 1

            # Select parents and perform crossover
            next_generation = []
            # for _ in range(self.population_size // 2):  # Pair up parents
            for _ in range(20):  # Pair up parents
                parent1 = Selection(population).select_parent(self.fitness_calculator)
                parent2 = Selection(population).select_parent(self.fitness_calculator)

                # Perform crossover
                crossover = Crossover(parent1, parent2)
                child1, child2 = crossover.cross()

                # Mutate children
                child1.mutate()
                child2.mutate()

                # Add children to the next generation
                next_generation.append(child1)
                next_generation.append(child2)

                print("Child Added!")
                # input()

            # Replace old population with the new generation
            population += next_generation
            print(len(population))
            # input("Then?")
            population = self.save_population(population)

            print(len(population))
            # input("Now?")

            self.save_images(population, gen)
            print(f"Generation {gen + 1} complete")
            # input("Continue?")

            self.fitnesses = {}

            # Save or check the best individual (elite) in the population
            # best_individual = min(population, key=lambda x: self.fitnesses[x])
            # print(f"Best Fitness: {best_individual.fitness}")

    def initialize_population(self):
        # TODO
        pop = []
        for _ in range(self.population_size):
            pop.append(
                DnaDrawing(height=canvas_height, width=canvas_width).get_random(
                    height=canvas_height, width=canvas_width
                )
            )
        return pop

    def save_population(self, drawings):
        # TODO
        # Depends on survivor selection scheme
        new_gen = []
        cutoff_fitness = list(reversed(sorted(self.fitnesses.values())))[self.population_size-1]
        print(cutoff_fitness)
        # input("CHECK")
        # print(sorted(self.fitnesses.values()))
        for i, individual in enumerate(drawings):
            if self.fitness_calculator.get_drawing_fitness(individual) >= cutoff_fitness:
                new_gen.append(individual)
                if len(new_gen) == self.population_size:
                    break
        return new_gen

    def save_images(self, drawings, generation):
        # Create directory if it does not exist
        if not os.path.exists("imgs"):
            os.makedirs("imgs")
        # Save images of the drawings
        for i, drawing in enumerate(drawings):
            if self.fitness_calculator.get_drawing_fitness(drawing) == min(self.fitnesses.values()):
                self.fitness_calculator.render_drawing(drawing).save(f"best_ones/gen_{generation}_drawing_{i}.png")
            # drawing.render(f"imgs/gen_{generation}_drawing_{i}.png")        

source_bitmap = "ml.bmp"
population_size = 100
generations = 100
ga = EvolutionEngine(source_bitmap, population_size, generations)
ga.evolve(generations)
