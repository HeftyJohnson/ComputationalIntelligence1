# Question 2 a and b
import operator
import random
import numpy
import math
from math import sin, sqrt
import matplotlib.pyplot as plt
from deap import base
from deap import creator
from deap import tools

posMinInit = - 500
posMaxInit = + 500
VMaxInit = 1.5
VMinInit = 0.5
populationSize = 50
dimension = 20
interval = 10
iterations = 400
maxnum = 2 ** 2

wmax = 0.9
wmin = 0.4
c1 = 2.0
c2 = 2.0

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, smin=None, smax=None, best=None)


def generate(size, smin, smax):
    part = creator.Particle(random.uniform(posMinInit, posMaxInit) for _ in range(size))
    part.speed = [random.uniform(VMinInit, VMaxInit) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part


def update_particle(part, best, weight):
    r1 = (random.uniform(0, 1) for _ in range(len(part)))
    r2 = (random.uniform(0, 1) for _ in range(len(part)))

    v_r0 = [weight * x for x in part.speed]
    v_r1 = [c1 * x for x in map(operator.mul, r1, map(operator.sub, part.best, part))]
    v_r2 = [c2 * x for x in map(operator.mul, r2, map(operator.sub, best, part))]

    part.speed = [0.7 * x for x in map(operator.add, v_r0, map(operator.add, v_r1, v_r2))]

    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)

    part[:] = list(map(operator.add, part, part.speed))


# def find_distance(vec1, vec2):
#     distance = 0.0
#     for i in range(len(vec1)):
#         distance += (vec1[i] - vec2[i]) ** 2
#     return math.sqrt(distance)
#
#
# def best_neighbour(pop, individual):
#     distances = list()
#     for sample in pop:
#         dist = find_distance(individual, sample)
#         distances.append((sample, dist))
#
#     distances.sort(key=lambda tup: tup[1])
#
#     best_n_dist = distances[0][0].fitness
#     best_n = distances[0][0]
#     for i in range(5):
#         if distances[i][0].fitness > best_n_dist:
#             best_n_dist = distances[i][0].fitness
#             best_n = distances[i][0]
#     return best_n


def eval_indv(individual):
    return -sum(x * sin(sqrt(abs(x))) for x in individual),


toolbox = base.Toolbox()
toolbox.register("particle", generate, size=dimension, smin=-3, smax=3)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", update_particle)
toolbox.register("evaluate", eval_indv)


def main():
    pop = toolbox.population(n=populationSize)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    best = None

    for g in range(iterations):
        w = wmax - (wmax - wmin) * g / iterations

        for part in pop:
            part.fitness.values = toolbox.evaluate(part)

            if (not part.best) or (part.best.fitness < part.fitness):
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values

            if (not best) or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values

        for part in pop:
            toolbox.update(part, best, w)

        if g % interval == 0:
            logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
            print(logbook.stream)

    print('best particle position is ', best)

    plt.figure(figsize=(20, 10))

    plt.plot(logbook.select('gen'), logbook.select('min'))

    plt.title("Fitness of Fittest Individual Over The Generations - Canonical", fontsize=20, fontweight='bold')
    plt.xlabel("Generations", fontsize=18, fontweight='bold')
    plt.ylabel("Fitness", fontsize=18, fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    plt.show()

    return pop, logbook, best


if __name__ == "__main__":
    main()
