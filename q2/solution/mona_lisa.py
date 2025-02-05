import random
import os
from PIL import image

minPoints = 3
active_polygons_min = 1

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

        origin = DnaPoint.get_random(width, height)
        for i in range(minPoints):
            pt = DnaPoint.get_random(width, height)
            pts.append(pt)

        brush = DnaBrush.get_random()

        return DnaPolygon(pts, brush)

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
        self.x = random.randint(0, width)
        self.y = random.randint(0, height)

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
        self.r = random.randint(0, 255)
        self.g = random.randint(0, 255)
        self.b = random.randint(0, 255)
        self.a = random.randint(0, 255)

    def clone(self):
        return DnaBrush(self.r, self.g, self.b, self.a)

# DnaDrawing class
class DnaDrawing:

    # Attributes
    width = 0
    height = 0
    is_dirty = False
    polygons = []

    def __init__(self, width, height, polygons, isDirty=True):
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
        for i in range(active_polygons_min):
            drawing.polygons.append(DnaPolygon.get_random(width, height))
        return drawing

    def clone(self):
        clonedPolygons = []
        for poly in polygons:
            clonedPolygons.append(poly.clone())
        return DnaDrawing(width, height, clonedPolygons, is_dirty)

    def mutate(self):
        # TODO
        pass

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
        self.source_bitmap = Image.open(source_bitmap)
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
        for shape in new_drawing.shapes:
            draw.polygon(shape.points, fill=shape.color)

        return rendered_image

# class NewFitnessCalculator:
#     source_bitmap = None
#     pixels = []

#     def __init__(self, source_bitmap):
#         self.source_bitmap = source_bitmap
#         # Borrowed from https://stackoverflow.com/a/1109747
#         im = Image.open(source_bitmap)
#         pixels = list(im.getdata())
#         width, height = im.size
#         pixels = [pixels[i * width:(i + 1) * width] for i in xrange(height)]

#     def get_drawing_fitness(self, drawing):
#         error = 0


# Depending on selection scheme, this may change
class Crossover:

    p1 = None
    p2 = None

    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def cross(self):
        # Determine random crossover point
        crossover_point = random.randint(0, min(len(p1.polygons), len(p2.polygons))-1)

        # Create children polygon lists
        child_one_polygons = self.p1.polygons[:crossover_point] + self.p2.polygons[crossover_point:]
        child_two_polygons = self.p2.polygons[:crossover_point] + self.p1.polygons[crossover_point:]

        child_one = DnaDrawing(self.p1.width, self.p1.height, child_one_polygons)
        child_two = DnaDrawing(self.p2.width, self.p2.height, child_two_polygons)

        return child_one, child_two

class Selection:

    def __init__(self, population):
        self.population = population

    def select_parent(self, drawings):
        # TODO
        # Change as per selection scheme

        # Roulette based drawing
        total_fitness = sum([individual.fitness for individual in self.population])

        # Pick a random fitness value and select the corresponding individual
        selection_value = random.uniform(0, total_fitness)
        cumulative_fitness = 0
        for individual in self.population:
            cumulative_fitness += individual.fitness
            if cumulative_fitness >= selection_value:
                return individual

class EvolutionEngine:
    source_bitmap = None
    population_size = 0
    fitness_calculator = None
    crossover = None
    selection = None
    generations = 0

    def __init__(self, source_bitmap, population_size, generations):
        self.source_bitmap = source_bitmap
        self.population_size = population_size
        self.generations = generations
        self.fitness_calculator = FitnessCalculator(source_bitmap)
    
    def evolve(self, generations):
        # TODO
        # Initialize population
        population = self.initialize_population()

        # Calculate fitness for each individual
        for gen in range(self.generations):
            print(f"Generation {gen + 1}")

            # Evaluate fitness of the population
            for individual in population:
                individual.fitness = self.fitness_calculator.get_drawing_fitness(individual)

            # Select parents and perform crossover
            next_generation = []
            for _ in range(self.population_size // 2):  # Pair up parents
                parent1 = Selection(population).select_parent()
                parent2 = Selection(population).select_parent()

                # Perform crossover
                crossover = Crossover(parent1, parent2)
                child1, child2 = crossover.cross()

                # Mutate children
                child1.mutate()
                child2.mutate()

                # Add children to the next generation
                next_generation.append(child1)
                next_generation.append(child2)

            # Replace old population with the new generation
            population = next_generation

            # Optionally: Save or check the best individual (elite) in the population
            best_individual = min(population, key=lambda x: x.fitness)
            print(f"Best Fitness: {best_individual.fitness}")


    def initialize_population(self):
        # TODO

    def save_population(self, drawings, generation):
        # TODO
