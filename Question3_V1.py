# question 3 1 - 6

import random

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy
from deap import base
from deap import creator
from deap import tools
from deap.benchmarks.tools import hypervolume
from deap.tools.emo import assignCrowdingDist
from deap.tools.emo import isDominated
from sympy.combinatorics.graycode import bin_to_gray, gray_to_bin

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()


def chrom2real(c):
    indasstring = ''.join(map(str, c))
    degray = gray_to_bin(indasstring)
    numasint = int(degray, 2)
    numinrange = -4 + 8 * numasint / (2 ** 10)
    return numinrange


def calcFitness(individual):
    x1_bits = individual[0:10]
    x2_bits = individual[10:20]
    x3_bits = individual[20:30]
    x1 = ("".join(str(i) for i in x1_bits))
    x2 = ("".join(str(i) for i in x2_bits))
    x3 = ("".join(str(i) for i in x3_bits))
    x1_gray = bin_to_gray(x1)
    x2_gray = bin_to_gray(x2)
    x3_gray = bin_to_gray(x3)
    x1 = chrom2real(x1_gray)
    x2 = chrom2real(x2_gray)
    x3 = chrom2real(x3_gray)
    f1 = (((x1 / 0.6) / 1.6) ** 2 + (x2 / 3.4) ** 2 + (x3 - 1.3) ** 2.0) / 2.0
    f2 = ((x1 / 1.9 - 2.3) ** 2 + (x2 / 3.3 - 7.1) ** 2 + (x3 + 4.3) ** 2.0) / 3.0
    return f1, f2


def separatevariables(v):
    return chrom2real(v[0:10]), chrom2real(v[10:20]), chrom2real(v[20:30])


def eval_fit(individual):
    seperated_indv = separatevariables(individual)
    f1 = (((seperated_indv[0] / 0.6) / 1.6) ** 2 + (seperated_indv[1] / 3.4) ** 2 + (
            seperated_indv[2] - 1.3) ** 2.0) / 2.0
    f2 = ((seperated_indv[0] / 1.9 - 2.3) ** 2 + (seperated_indv[1] / 3.3 - 7.1) ** 2 + (
            seperated_indv[2] + 4.3) ** 2.0) / 3.0
    return f1, f2


def algorithm(indivdual, fronts):
    x = len(fronts)
    k = 1

    while True:
        dominated = False
        for indv in reversed(fronts[k - 1]):
            if isDominated(indv.fitness.values, indivdual.fitness.values):
                dominated = True
                break

        if not dominated:
            fronts[k - 1].append(indivdual)
            return k
            break

        else:
            k = k + 1
            if k > x:
                fronts.append([indivdual])
                return x + 1
                break


def efficient_non_dominated_sort(pop_sorted_by_f1):
    front = [[]]

    q = pop_sorted_by_f1[0]
    front[0].append(q)

    for ind in pop_sorted_by_f1[1:]:
        algorithm(ind, front)

    return front


def take_first(elem):
    return elem[0]


toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, 30)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", calcFitness)
toolbox.register("mate", tools.cxUniform, indpb=0.9)
flipProb = 1.0 / 30
toolbox.register("mutate", tools.mutFlipBit, indpb=flipProb)
toolbox.register("select", tools.selNSGA2)


def main(seed=None):
    random.seed(seed)

    total_population = 24

    pop = toolbox.population(n=total_population)

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pop = toolbox.select(pop, len(pop))

    # code for 3.1

    for individual in pop:
        x_values = separatevariables(individual)
        fitness_values = eval_fit(individual)
        x1 = x_values[0]
        x2 = x_values[1]
        x3 = x_values[2]
        f1 = fitness_values[0]
        f2 = fitness_values[1]
        print(x1, x2, x3, f1, f2)

    # code for 3.2

    pop.sort(key=lambda x: x.fitness.values[1])

    worstF2 = pop[-1].fitness.values[1]

    pop.sort(key=lambda x: x.fitness.values[0])

    worstF1 = pop[-1].fitness.values[0]

    fronts = efficient_non_dominated_sort(pop)

    for ind in pop:
        i = 0
        needs_home = True
        while needs_home:
            if ind in fronts[i]:
                needs_home = False
                ind.front = i + 1
            else:
                i += 1

    pop.sort(key=lambda x: x.front)

    for individual in pop:
        print(individual.fitness.values[0], individual.fitness.values[1], individual.front)

    plt.figure(figsize=(10, 6))

    i = 0
    front_f1 = []
    front_f2 = []
    while i != len(fronts):
        for ind in fronts[i]:
            front_f1.append(ind.fitness.values[0])
            front_f2.append(ind.fitness.values[1])
        plt.plot(front_f1, front_f2, marker='o', label='front ' + str(i + 1))
        i += 1
        front_f1.clear()
        front_f2.clear()

    plt.title("Plotting of Fronts", fontsize=20, fontweight='bold')
    plt.xlabel("F1 values", fontsize=18, fontweight='bold')
    plt.ylabel("F2 values", fontsize=18, fontweight='bold')
    plt.legend(loc='best')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.show()

    # code for 3.3

    for front in fronts:
        assignCrowdingDist(front)

    for individual in pop:
        print(individual.fitness.values[0], individual.fitness.values[1], individual.front,
              individual.fitness.crowding_dist)

    # code for 3.4

    offspring = tools.selTournamentDCD(pop, len(pop))
    offspring = [toolbox.clone(ind) for ind in offspring]

    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        toolbox.mate(ind1, ind2)

        toolbox.mutate(ind1)
        toolbox.mutate(ind2)
        del ind1.fitness.values, ind2.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    f1_off_values = []
    f2_off_values = []
    for ind in offspring:
        f1_off_values.append(ind.fitness.values[0])
        f2_off_values.append(ind.fitness.values[1])

    f1_pop_values = []
    f2_pop_values = []
    for ind in pop:
        f1_pop_values.append(ind.fitness.values[0])
        f2_pop_values.append(ind.fitness.values[1])

    plt.figure(figsize=(10, 6))
    plt.xlabel("F1 Values", fontsize=18, fontweight='bold')
    plt.ylabel("F2 Values", fontsize=18, fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    f1_blue = mpatches.Patch(color='blue', label='Offspring')
    f2_red = mpatches.Patch(color='red', label='Parents')

    plt.legend(handles=[f1_blue, f2_red])

    plt.scatter(f1_off_values, f2_off_values, color='blue')
    plt.scatter(f1_pop_values, f2_pop_values, color='red')

    plt.show()

    # code for 3.5

    combined_individuals = pop + offspring
    combined_individuals.sort(key=lambda x: x.fitness.values[0])
    fronts_combined = efficient_non_dominated_sort(combined_individuals)

    for front in fronts_combined:
        assignCrowdingDist(front)
        front.sort(key=lambda x: x.fitness.crowding_dist, reverse=True)

    i = 24
    front = 0
    indv = 0
    new_pop = []
    while i != 0:
        if indv == len(fronts_combined[front]):
            front += 1
            indv = 0
        new_pop.append(fronts_combined[front][indv])
        indv += 1
        i -= 1

    f1_new_values = []
    f2_new_values = []
    for ind in new_pop:
        f1_new_values.append(ind.fitness.values[0])
        f2_new_values.append(ind.fitness.values[1])

    f1_combined_values = []
    f2_combined_values = []
    for ind in combined_individuals:
        f1_combined_values.append(ind.fitness.values[0])
        f2_combined_values.append(ind.fitness.values[1])

    plt.figure(figsize=(10, 6))
    plt.xlabel("F1 Values", fontsize=18, fontweight='bold')
    plt.ylabel("F2 Values", fontsize=18, fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    f1_blue = mpatches.Patch(color='blue', label='New Population')
    f2_red = mpatches.Patch(color='red', label='Rejects')

    plt.legend(handles=[f1_blue, f2_red])

    plt.scatter(f1_combined_values, f2_combined_values, color='red')
    plt.scatter(f1_new_values, f2_new_values, color='blue')

    plt.show()

    # code for 3.6

    pop = new_pop

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    indvs_in_front_1 = []
    for x in pop:
        if x.front == 1:
            indvs_in_front_1.append(x)

    hypervol = [hypervolume(indvs_in_front_1, [worstF1, worstF2])]

    for gen in range(1, 30):
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        combined_individuals = pop + offspring
        combined_individuals.sort(key=lambda x: x.fitness.values[0])
        fronts_combined = efficient_non_dominated_sort(combined_individuals)

        for front in fronts_combined:
            assignCrowdingDist(front)
            front.sort(key=lambda x: x.fitness.crowding_dist, reverse=True)

        i = 24
        front = 0
        indv = 0
        new_pop = []
        while i != 0:
            if indv == len(fronts_combined[front]):
                front += 1
                indv = 0
            new_pop.append(fronts_combined[front][indv])
            indv += 1
            i -= 1

        pop = new_pop

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)

        indvs_in_front_1 = []
        for x in pop:
            if x.front == 1:
                indvs_in_front_1.append(x)

        hypervol.append(hypervolume(indvs_in_front_1, [worstF1, worstF2]))
        print(logbook.stream)

    plt.figure(figsize=(10, 6))
    plt.plot(logbook.select('gen'), hypervol)
    plt.title("Hypervolume of Non-Dominated Individuals over the generations", fontsize=20, fontweight='bold')
    plt.xlabel("Generations", fontsize=18, fontweight='bold')
    plt.ylabel("Hypervolume", fontsize=18, fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.show()

    f1_values = []
    f2_values = []
    for ind in pop:
        f1_values.append(ind.fitness.values[0])
        f2_values.append(ind.fitness.values[1])

    plt.figure(figsize=(10, 6))
    plt.scatter(f1_values, f2_values, color='blue')
    plt.scatter(worstF1, worstF2, color="red")
    f1_blue = mpatches.Patch(color='blue', label='Final Population')
    f2_red = mpatches.Patch(color='red', label='Reference Point')
    plt.legend(handles=[f1_blue, f2_red])
    plt.title("Final Population with Reference Point", fontsize=20, fontweight='bold')
    plt.xlabel("F1 values", fontsize=18, fontweight='bold')
    plt.ylabel("F2 values", fontsize=18, fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.show()


if __name__ == "__main__":
    main()
