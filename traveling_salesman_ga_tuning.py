# import mlrose
from mlrose import mlrose
import numpy as np
import matplotlib.pyplot as plt
import time
from util import plot_param_curve

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
pop = [50, 100, 150, 200, 250, 300, 500, 1000, 2000]
# ________________________________________________________________________________________________________________________
times_ga = []
fitns_ga = []
# Solve problem using the genetic algorithm
for i in pop:
    start = time.time()
    best_state, best_fitness, c = mlrose.genetic_alg(problem_fit,  pop_size=i, mutation_prob=0.1,
                                                        curve=True, max_iters=10, random_state=3)

    end = time.time()   
    print('genetic algorithm @ {} population'.format(i))
    print(best_fitness)
    print(end-start)
    print(len(c))
    times_ga.append(end-start)
    fitns_ga.append(best_fitness)
# ________________________________________________________________________________________________________________________

# ________________________________________________________________________________________________________________________

plot_param_curve(x_axis=pop, curve=fitns_ga, title='Fitness', param='Population')
plt.savefig('Figs/tsp-fit-ga-pop-tune')

mut = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
# ________________________________________________________________________________________________________________________
times_ga = []
fitns_ga = []
# Solve problem using the genetic algorithm
for i in mut:
    start = time.time()
    best_state, best_fitness, c = mlrose.genetic_alg(problem_fit,  pop_size=200, mutation_prob=i,
                                                        curve=True, max_iters=10, random_state=3)

    end = time.time()   
    print('genetic algorithm @ {} mutation'.format(i))
    print(best_fitness)
    print(end-start)
    print(len(c))
    times_ga.append(end-start)
    fitns_ga.append(best_fitness)
# ________________________________________________________________________________________________________________________

# ________________________________________________________________________________________________________________________

plot_param_curve(x_axis=mut, curve=fitns_ga, title='Fitness', param='Mutation')
plt.savefig('Figs/tsp-fit-ga-mut-tune')