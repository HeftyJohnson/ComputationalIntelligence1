# Question 2 c

import random
import numpy
import matplotlib.pyplot as plt
import math
from math import sin, sqrt
from deap import base
from deap import creator
from deap import tools

posMinInit = -500
posMaxInit = +500
VMaxInit = 1.5
VMinInit = 0.5
dimension = 20
interval = 10
iterations = 400
populationSize = 50

epsilon = dimension / 100.0 * 0.01


def getcenter(pop):
    center = list()
    for j in range(dimension):
        centerj = 0
        for i in pop:
            centerj += i[j]
        centerj /= populationSize
        center.append(centerj)
    return center


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, smin=None, smax=None, best=None)


def generate(size, smin, smax):
    part = creator.Particle(random.uniform(posMinInit, posMaxInit) for _ in range(size))
    part.speed = [random.uniform(VMinInit, VMaxInit) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part


def updateParticle(part, pop, center, i):
    r1 = random.uniform(0, 1)
    r2 = random.uniform(0, 1)
    r3 = random.uniform(0, 1)

    demonstrator = random.choice(list(pop[0:i]))

    for j in range(dimension):
        part.speed[j] = r1 * part.speed[j] + r2 * (demonstrator[j] - part[j]) + r3 * epsilon * (center[j] - part[j])
        part[j] = part[j] + part.speed[j]
        if part[j] < -posMinInit or part[j] > posMaxInit:
            part[j] = random.uniform(posMinInit, posMaxInit)

def eval_indv(individual):
    return -sum(x * sin(sqrt(abs(x))) for x in individual),

toolbox = base.Toolbox()
toolbox.register("particle", generate, size=dimension, smin=-3, smax=3)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle)
toolbox.register("evaluate", eval_indv)


def main():
    pop = toolbox.population(n=populationSize)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    prob = [0] * populationSize
    for i in range(len(pop)):
        prob[populationSize - i - 1] = 1 - i / (populationSize - 1)
        prob[populationSize - i - 1] = pow(prob[populationSize - i - 1],
                                           math.log(math.sqrt(math.ceil(dimension / 100.0))))

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    for g in range(iterations):

        for part in pop:
            part.fitness.values = toolbox.evaluate(part)

        pop.sort(key=lambda x: x.fitness, reverse=True)

        center = getcenter(pop)

        for i in reversed(range(len(pop) - 1)):

            if random.uniform(0, 1) < prob[i + 1]:
                toolbox.update(pop[i + 1], pop, center, i + 1)

        if g % interval == 0:
            logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
            print(logbook.stream)

    print(pop[0])

    plt.figure(figsize=(20, 10))

    plt.plot(logbook.select('gen'), logbook.select('min'))

    plt.title("Fitness of Fittest Individual Over The Generations - Social Learning", fontsize=20, fontweight='bold')
    plt.xlabel("Generations", fontsize=18, fontweight='bold')
    plt.ylabel("Fitness", fontsize=18, fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    plt.show()

    return pop, logbook


if __name__ == "__main__":
    main()
