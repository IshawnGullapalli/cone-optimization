import random
import numpy
import matplotlib.pyplot as plt
from math import pi

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

# constants
MAX_ATTRIBUTE_SIZE = 25

# two objectives: maximize volume, minimize surface area
creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
# each individual is a list [radius, height]
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("attr_item", random.randint, 1, MAX_ATTRIBUTE_SIZE)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_item, 2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def calculate_volume(r, h):
    """
    This function calculates the volume of a cone.

    :param r: radius
    :param h: height
    :return: volume of cone
    """
    return pi * (r ** 2) * (h / 3)


def calculate_surface_area(r, h):
    """
    This function calculates the surface area of a cone.

    :param r: radius
    :param h: height
    :return: surface area of cone
    """
    return pi * r * (r + ((h ** 2) + (r ** 2)) ** 0.5)


def evaluate_fitness(individual):
    """
    This function calculates the fitness score of an individual. The fitness score is simply a tuple containing the
    volume and surface area of a cone. The function also plots the current individual.

    :param individual: the current individual
    :return: a tuple (volume, surface area)
    """
    r, h = individual[0], individual[1]
    volume = calculate_volume(r, h)
    surface_area = calculate_surface_area(r, h)
    plt.plot(volume, surface_area, '.', color="black")
    return volume, surface_area


def crossover(ind1, ind2):
    """
    This function returns two children made from the two individuals passed in as arguments. The children are found by
    swapping the values of the two parents.

    :param ind1: the first parent
    :param ind2: the second parent
    :return: a tuple containing two individuals representing the children
    """
    r1 = ind1[0]
    h1 = ind1[1]
    ind1[0] = ind2[0]
    ind2[0] = r1
    ind1[1] = ind2[1]
    ind2[0] = h1
    return ind1, ind2


def main(population_size, max_generations):
    """
    This function is the main genetic algorithm. It generates various elements until the target phrase is reached.
    Every generation, it prints relevant information to the terminal. Once it terminates, it will have found a Pareto
    optimal set of items.

    :param population_size: the size of the population
    :param max_generations: the number of generations before terminating
    :return the population, statistics, and the best individuals
    """
    toolbox.register("evaluate", evaluate_fitness)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=MAX_ATTRIBUTE_SIZE, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)

    NGEN = max_generations
    MU = population_size
    LAMBDA = 100
    CXPB = 0.7
    MUTPB = 0.2

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof)

    return pop, stats, hof


"""
This executes the genetic algorithm with the following parameters:
- Population size: 500
- Max Generations: 200
These values can be modified. After the genetic algorithm terminates, we print and plot every Pareto optimal individual.
"""
if __name__ == '__main__':
    pop, stats, hof = main(500, 200)
    x = []
    y = []
    for ind in hof:
        x.append(calculate_volume(ind[0], ind[1]))
        y.append(calculate_surface_area(ind[0], ind[1]))

    plt.plot(x, y, '.', color="green")
    plt.ticklabel_format(style="plain")
    plt.xlabel("Volume")
    plt.ylabel("Surface Area")
    plt.savefig('objective_space.png')
