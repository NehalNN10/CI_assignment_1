import random
from PIL import Image, ImageDraw
from colour.difference import delta_E_CIE1976
import numpy as np

MIN_POINTS = 3
MAX_POINTS = 10
POLYGONS = 50

class Chromosome:
    def __init__(self, length, width):
        self.length = length
        self.width = width
        self.fitness = float('inf') # Objective is to minimize fitness
        self.array = None
        self.image = None
        self.gen_image()

    def random_color(self):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    def random_bg(self):
        self.image = Image.new('RGBA', (self.length, self.width), self.random_color())

    def gen_image(self):
        region = (self.length + self.width) // 8 # ? what?

        img = Image.new('RGBA', (self.length, self.width), self.random_color())

        for i in range(POLYGONS):
            no_vertices = random.randint(MIN_POINTS, MAX_POINTS)

            region_x = random.randint(0, self.length)
            region_y = random.randint(0, self.width)

            xy = []
            for j in range(no_vertices):
                x = random.randint(region_x - region, region_x + region)
                y = random.randint(region_y - region, region_y + region)
                xy.append((x, y))

            img1 = ImageDraw.Draw(img) # For background
            img1.polygon(xy, fill=self.random_color()) # Add polygons

        self.image = img
        self.array = self.to_array(img)

    def create_random_image_array_2(self):
        self.array = np.random.randint(0, 255, (self.length, self.width, 4)).astype(np.uint8)

        self.image = Image.fromarray(self.array.astype(np.uint8))

    def add_shape(self):
        region = random.randint(1, (self.length + self.width) // 4)

        img = self.image

        no_vertices = random.randint(MIN_POINTS, MAX_POINTS)
        region_x = random.randint(0, self.length)
        region_y = random.randint(0, self.width)
        xy = []
        for j in range(no_vertices):
            x = random.randint(region_x - region, region_x + region)
            y = random.randint(region_y - region, region_y + region)
            xy.append((x, y))
        
        img1 = ImageDraw.Draw(img)
        img1.polygon(xy, fill=self.random_color())
    
        self.image = img
        self.array = self.to_array(img)
    
    def to_image(self):
        im = Image.fromarray(self.array)
        im.show()
        return im

    def to_array(self, img):
        return np.array(img)

    def get_color_fitness(self, target):
        self.fitness = np.mean(delta_E_CIE1976(target, self.array))

    def get_fitness_euclidean(self, target):
        diff_array = np.subtract(target, self.array)
        self.fitness = np.mean(np.absolute(diff_array))