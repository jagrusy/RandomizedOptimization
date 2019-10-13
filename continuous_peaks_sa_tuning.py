# import mlrose
from mlrose import mlrose
import numpy as np
import matplotlib.pyplot as plt
import time
from util import plot_temp_curve

np.random.seed(seed=0)

# Initialize fitness function object
fitness_fn = mlrose.ContinuousPeaks(t_pct=0.1)

state = np.random.randint(low=0, high=2, size=1000)

# Define optimization problem object
problem_fit = mlrose.DiscreteOpt(length=len(state), fitness_fn=fitness_fn,
                                                    maximize=True, max_val=2)
temps = [0.1, 1.0, 5.0, 10.0, 100.0]
# Define decay schedule
times_sa_exp = []
fitns_sa_exp = []
# Solve problem using simulated annealing
for i in temps:
    schedule = mlrose.ExpDecay(init_temp=i, exp_const=0.005, min_temp=0.005)
    start = time.time()
    best_state, best_fitness, c = mlrose.simulated_annealing(problem_fit, schedule=schedule,
                                                        max_attempts=10, max_iters=10000,
                                                        init_state=None, curve=True,
                                                        random_state=3)
    end = time.time()
    print('sa algorithm @ {} exponential'.format(i))
    print(best_fitness)
    print(end-start)
    print(len(c))
    times_sa_exp.append(end-start)
    fitns_sa_exp.append(best_fitness)
# ________________________________________________________________________________________________________________________
# Solve problem using simulated annealing
times_sa_art = []
fitns_sa_art = []
for i in temps:
    schedule = mlrose.ArithDecay(init_temp=i, decay=0.0001*i, min_temp=0.005)
    start = time.time()
    best_state, best_fitness, c = mlrose.simulated_annealing(problem_fit, schedule=schedule,
                                                        max_attempts=10, max_iters=10000,
                                                        init_state=None, curve=True,
                                                        random_state=3)
    end = time.time()
    print('sa algorithm @ {} arithmetic'.format(i))
    print(best_fitness)
    print(end-start)
    print(len(c))
    times_sa_art.append(end-start)
    fitns_sa_art.append(best_fitness)
# ________________________________________________________________________________________________________________________

plot_temp_curve(x_axis=temps, curve_exp=fitns_sa_exp, curve_art=fitns_sa_art, title='Fitness')
plt.savefig('Figs/con-peaks-fit-sa-tuning')
