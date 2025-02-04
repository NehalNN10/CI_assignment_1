import random
import os
from PIL import image

# Class for polygons
class DnaPolygon:
    points = []
    brush = None

    def __init__(self, points, brush):
        self.points = points
        self.brush = brush

    def get_random(self, width, height):
        # TODO

    def clone(self):
        return DnaPolygon(self.points, self.brush)

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

    def __init__(self, width, height, polygons):
        self.width = width
        self.height = height
        self.polygons = polygons

    def mutate(self):
        # TODO

class NewFitnessCalculator:
    source_bitmap = None

    def __init__(self, source_bitmap):
        self.source_bitmap = source_bitmap

    def get_drawing_fitness(self, drawing):
        # TODO

class Crossover:
    def cross(self, dna_drawing1, dna_drawing2):
        # TODO

class Selection:
    def select(self, drawings):
        # TODO

class EvolutionEngine:
    source_bitmap = None
    population_size = 0
    fitness_calculator = None
    crossover = None
    selection = None

    def __init__(self, source_bitmap, population_size):
        self.source_bitmap = source_bitmap
        self.population_size = population_size
    
    def evolve(self, generations):
        # TODO

    def initialize_population(self):
        # TODO

    def save_population(self, drawings, generation):
        # TODO
