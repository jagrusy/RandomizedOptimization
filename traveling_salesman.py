# import mlrose
from mlrose import mlrose
import numpy as np
import matplotlib.pyplot as plt
import time
from util import plot_fitness_curve

np.random.seed(seed=0)
# Create list of city coordinates
# coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]
coords_list = []
for i in range(100):
    coords_list.append((np.random.randint(0, 100), np.random.randint(0, 100)))
x = [x[0] for x in coords_list]
y = [x[1] for x in coords_list]

fig, ax = plt.subplots()
ax.grid()
ax.scatter(x, y)

for i in range(0, len(coords_list)):
    # Number the coordinates on the map
    ax.annotate(i, (x[i]+0.1, y[i]+0.1))
plt.savefig('Figs/coordinate-plot')

# Initialize fitness function object using coords_list
fitness_coords = mlrose.TravellingSales(coords=coords_list)

# Define optimization problem object
problem_fit = mlrose.TSPOpt(length=len(coords_list), fitness_fn=fitness_coords, maximize=False)
iterations = [1, 10, 100]
# ________________________________________________________________________________________________________________________
times_ga = []
fitns_ga = []
# Solve problem using the genetic algorithm
for i in iterations:
    start = time.time()
    best_state, best_fitness, c = mlrose.genetic_alg(problem_fit,  pop_size=200, mutation_prob=0.1,
                                                        curve=True, max_iters=i, random_state=3)

    end = time.time()   
    print('genetic algorithm @ {} iterations'.format(i))
    print(best_fitness)
    print(end-start)
    print(len(c))
    times_ga.append(end-start)
    fitns_ga.append(best_fitness)
# ________________________________________________________________________________________________________________________
# Define decay schedule
schedule = mlrose.ExpDecay()
times_sa = []
fitns_sa = []
# Solve problem using simulated annealing
for i in iterations:
    start = time.time()
    best_state, best_fitness, c = mlrose.simulated_annealing(problem_fit, schedule=schedule,
                                                        max_attempts=10, max_iters=i,
                                                        init_state=None, curve=True,
                                                        random_state=3)
    end = time.time()
    print('sa algorithm @ {} iterations'.format(i))
    print(best_fitness)
    print(end-start)
    print(len(c))
    times_sa.append(end-start)
    fitns_sa.append(best_fitness)
# ________________________________________________________________________________________________________________________
# Solve problem using random hill climb
times_rhc = []
fitns_rhc = []
for i in iterations:
    start = time.time()
    best_state, best_fitness, c = mlrose.random_hill_climb(problem_fit, max_attempts=10,
                                                        max_iters=i, restarts=0,
                                                        init_state=None, curve=True,
                                                        random_state=3)

    end = time.time()
    print('rhc algorithm @ {} iterations'.format(i))
    print(best_fitness)
    print(end-start)
    print(len(c))
    times_rhc.append(end-start)
    fitns_rhc.append(best_fitness)
# ________________________________________________________________________________________________________________________
# Solve using MIMIC
times_mim = []
fitns_mim = []
for i in iterations:
    start = time.time()
    # if population size < 200, there are not enough specimen to improve after mutation and crossover
    best_state, best_fitness, c = mlrose.mimic(problem_fit, pop_size=200, keep_pct=0.25, 
                                                        max_attempts=10, max_iters=i,
                                                        curve=True, random_state=3,
                                                        fast_mimic=True)

    end = time.time()
    print('MIMIC algorithm @ {} iterations'.format(i))
    print(best_fitness)
    print(end-start)
    print(len(c))
    times_mim.append(end-start)
    fitns_mim.append(best_fitness)
# ________________________________________________________________________________________________________________________

plot_fitness_curve(x_axis=iterations, curve_rhc=fitns_rhc, curve_sa=fitns_sa, curve_ga=fitns_ga, curve_mim=fitns_mim, title='Fitness')
plt.savefig('Figs/tsp-fit')
plot_fitness_curve(x_axis=iterations, curve_rhc=times_rhc, curve_sa=times_sa, curve_ga=times_ga, curve_mim=times_mim, title='Timing')
plt.savefig('Figs/tsp-timing')