# import mlrose
from mlrose import mlrose
import numpy as np
import matplotlib.pyplot as plt
import time
from util import plot_fitness_curve

np.random.seed(seed=0)

# Initialize fitness function object
fitness_fn = mlrose.FourPeaks(t_pct=0.1)

state = np.random.randint(low=0, high=2, size=1000)

# Define optimization problem object
problem_fit = mlrose.DiscreteOpt(length=len(state), fitness_fn=fitness_fn,
                                                    maximize=True, max_val=2)
iterations = [1, 10, 100, 1000, 10000]
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
                                                        max_iters=i, restarts=1,
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
    best_state, best_fitness, c = mlrose.mimic(problem_fit, pop_size=200, keep_pct=0.4, 
                                                        max_attempts=5, max_iters=i,
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
plt.savefig('Figs/4-peaks-fit')
plot_fitness_curve(x_axis=iterations, curve_rhc=times_rhc, curve_sa=times_sa, curve_ga=times_ga, curve_mim=times_mim, title='Timing')
plt.savefig('Figs/4-peaks-timing')