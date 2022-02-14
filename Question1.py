# Question 1

import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sympy.combinatorics.graycode import gray_to_bin
from deap import creator, base, tools

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

popSize = 50
dimension = 2
numOfBits = 15
iterations = 100
dspInterval = 10
nElitists = 1
omega = 5
crossPoints = 2
crossProb = 0.6
flipProb = 1. / (dimension * numOfBits)
mutateprob = .1
maxnum = 2 ** numOfBits

individual_best = []
overall_best = []

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)

toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, numOfBits * dimension)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def chrom2real(c):
    indasstring = ''.join(map(str, c))
    degray = gray_to_bin(indasstring)
    numasint = int(degray, 2)
    numinrange = -5 + 10 * numasint / maxnum
    return numinrange


def separate_variables(v):
    return chrom2real(v[0:numOfBits]), chrom2real(v[numOfBits:])


# f (x1, x2) = 2 + 4.1(x1)^2 - 2.1(x1)^4 + 1/3(x1)^6 + (x1)(x2) - 4((x2)-0.05)^2 + 4(x2)^4
def eval_fit(individual):
    sep = separate_variables(individual)
    f = (2 + (4.1 * (sep[0] ** 2)) - (2.1 * (sep[0] ** 4)) + ((1 / 3) * (sep[0] ** 6)) + (sep[0] * sep[1])
         - (4 * ((sep[1] - 0.05) ** 2)) + (4 * (sep[1] ** 4)))
    return 1.0 / (0.01 + f),


def convert_fitness(fitness):
    return (1 / fitness) - 0.01


def f(x1, x2):
    return (2 + (4.1 * (x1 ** 2)) - (2.1 * (x1 ** 4)) + ((1 / 3) * (x1 ** 6)) + (x1 * x2)
            - (4 * ((x2 - 0.05) ** 2)) + (4 * (x2 ** 4)))


def dx1(x1, x2):
    return 2 * x1 ** 5 - 8.4 * x1 ** 3 + 8.2 * x1 + x2


def dx2(x1, x2):
    return x1 + 16 * x2 ** 3 - 8 * x2 + 0.4


toolbox.register("evaluate", eval_fit)

toolbox.register("mate", tools.cxTwoPoint)

toolbox.register("mutate", tools.mutFlipBit, indpb=flipProb)

toolbox.register("select", tools.selRoulette, fit_attr='fitness')

hall_of_fame = tools.HallOfFame(1)

stats = tools.Statistics()

stats.register('Min', np.min)
stats.register('Max', np.max)
stats.register('Avg', np.mean)
stats.register('Std', np.std)

logbook = tools.Logbook()


def main():
    pop = toolbox.population(n=popSize)

    fitnesses = list(map(toolbox.evaluate, pop))

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    g = 0

    hall_of_fame.clear()

    while g < iterations:

        g = g + 1
        print("-- Generation %i --" % g)

        offspring = tools.selBest(pop, nElitists) + toolbox.select(pop, len(pop) - nElitists)

        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            if random.random() < crossProb:
                toolbox.mate(child1, child2)

                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            if random.random() < mutateprob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        this_gen_fitness = []
        for ind in offspring:
            this_gen_fitness.append(ind.fitness.values[0])

        hall_of_fame.update(offspring)

        stats_of_this_gen = stats.compile(this_gen_fitness)

        stats_of_this_gen['Generation'] = g

        logbook.append(stats_of_this_gen)

        pop[:] = offspring

        individual_best.append(convert_fitness(tools.selBest(pop, 1)[0].fitness.values[0]))
        overall_best.append(tools.selBest(pop, 1)[0])

        if g % dspInterval == 0:
            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5

            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    print("Decoded x1, x2 is %s, %s" % (separate_variables(best_ind)))

    # plt.figure(figsize=(20, 10))
    #
    # plt.plot(logbook.select('Generation'), logbook.select('Min'))
    #
    # plt.title("Fitness of Fittest Individual Over The Generations", fontsize=20, fontweight='bold')
    # plt.xlabel("Generations", fontsize=18, fontweight='bold')
    # plt.ylabel("Fitness", fontsize=18, fontweight='bold')
    # plt.xticks(fontweight='bold')
    # plt.yticks(fontweight='bold')

    plt.figure(figsize=(20, 10))
    plt.plot(logbook.select('Generation'), individual_best)
    plt.title("Fitness of Fittest Individual At End Of Generations", fontsize=20, fontweight='bold')
    plt.xlabel("Generations", fontsize=18, fontweight='bold')
    plt.ylabel("Fitness", fontsize=18, fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    xrange = np.linspace(-2.1, 2.1, 100)
    yrange = np.linspace(-1.1, 1.1, 100)
    X, Y = np.meshgrid(xrange, yrange)
    Z = f(X, Y)

    fig = plt.figure(figsize=(10, 6))

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False,
                        zorder=0)

    fig.colorbar(p, shrink=0.5)

    x1list = []
    x2list = []
    zlist = []

    for x in range(len(overall_best)):
        decision_variables = separate_variables(overall_best[x])
        x1 = decision_variables[0]
        x2 = decision_variables[1]
        x1list.append(x1)
        x2list.append(x2)
        z = f(x1, x2)
        zlist.append(z)
    ax.plot3D(x1list, x2list, zlist, color="k", marker="o", zorder=10)

    decision_variables = separate_variables(overall_best[len(overall_best) - 1])
    x1 = decision_variables[0]
    x2 = decision_variables[1]
    ax.plot3D([x1], [x2], [f(x1, x2)], color="#FF0000", marker="o", zorder=10)
    ax.view_init(80, 30)

    plt.show()

    # 1.2 code

    x1 = 1
    x2 = 3

    xlist = []
    ylist = []
    zlist = []
    alpha = 0.01

    for step in range(0, 200):
        x1 = x1 - alpha * (dx1(x1, x2))
        x2 = x2 - alpha * (dx2(x1, x2))
        z = f(x1, x2)
        xlist.append(x1)
        ylist.append(x2)
        zlist.append(z)

    x = xlist[-1]
    y = ylist[-1]
    z = zlist[-1]

    fig = plt.figure(figsize=(10, 10))

    # surface_plot with color grading and color bar
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False,
                    zorder=0)
    ax.plot3D(xlist, ylist, zlist, color="k", marker='o', zorder=10)
    ax.plot3D(x, y, z, color='red', marker='o', zorder=10)
    ax.view_init(80, 30)

    plt.show()

    x1 = 3
    x2 = 2

    xlist = []
    ylist = []
    zlist = []
    alpha = 0.01

    for step in range(0, 200):
        x1 = x1 - alpha * (dx1(x1, x2))
        x2 = x2 - alpha * (dx2(x1, x2))
        z = f(x1, x2)
        xlist.append(x1)
        ylist.append(x2)
        zlist.append(z)

    x = xlist[-1]
    y = ylist[-1]
    z = zlist[-1]

    fig = plt.figure(figsize=(10, 10))

    # surface_plot with color grading and color bar
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False,
                    zorder=0)
    ax.plot3D(xlist, ylist, zlist, color="k", marker='o', zorder=10)
    ax.plot3D(x, y, z, color='red', marker='o', zorder=10)
    ax.view_init(80, 30)

    plt.show()


if __name__ == '__main__':
    main()
