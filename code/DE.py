from problem import GymProblem
import numpy as np
import matplotlib.pyplot as plt

class DifferentialEvolution:

    def __init__(self, population_size, F, cr, max_iterations, cr_strategy):
        self.population_size = population_size    
        self.F = F
        self.cr = cr
        self.max_iterations = max_iterations
        self.cr_strategy = cr_strategy
        self.FE_max = 1000
        self.fitness_history = []
        self.solution_history = []
        self.reward_history = []
        self.FE = 0
        self.FE_history = []

    # DE/rand/1 
    def mutate(self, population, individual_idx):
        idxs = [idx for idx in range(len(population)) if idx != individual_idx]
        r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
        return population[r1] + self.F*(population[r2]-population[r3])

    # DE/rand/1/bin
    def crossover_bin(self, population, mutants):  
        trials = np.zeros_like(population)
        pop_size, n = population.shape

        for i in range(pop_size):
            # Value to guarantee at least one value comes from the mutant
            value = np.random.randint(0, n)
            for j in range(n):
                if np.random.rand() < self.cr or j == value:
                    trials[i, j] = mutants[i, j]
                else:
                    trials[i, j] = population[i, j]
        return trials

    # DE/rand/1/exp
    def crossover_exp(self, population, mutants):
        trials = population.copy()
        pop_size, D = population.shape

        for i in range(pop_size):
            k = np.random.randint(0, D)
            L = 0
            while np.random.rand() < self.cr and L < D:
                L += 1

            for n in range(L):
                j = (k+n) % D
                trials[i, j] = mutants[i, j]
        return trials

    # probability-based search
    def crossover_p(self, population, mutants, FE):
        trials = np.zeros_like(population)

        p_min = 0.1
        p_max = 0.9
        p = p_min + (p_max-p_min) * (FE/self.FE_max)

        for i in range(self.population_size):
            if np.random.rand() > p:
                trials[i] = self.crossover_bin(population=population[i:i+1], mutants=mutants[i:i+1])[0]
            else:
                trials[i] = self.crossover_exp(population=population[i:i+1], mutants=mutants[i:i+1])[0]
        return trials

    def selection(self, fitnesses, trial_fitnesses):
        improved =  trial_fitnesses > fitnesses
        return improved

    def __call__(self, problem):
        population = np.array([problem.sample() for _ in range(self.population_size)])
        fitnesses = np.zeros(self.population_size)
        rewards = []
        fitness_hist = []

        for i, individual in enumerate(population):
            fitness, rewards_ind = problem.play_episode(individual)
            fitnesses[i] = fitness
            rewards.append(rewards_ind)
            self.FE += 1

        idx_prime = np.argmax(fitnesses)
        iteration = 0

        for _ in range(self.max_iterations): 
            if self.FE > self.FE_max:
              break
            mutants = np.zeros_like(population)

            for i in range(self.population_size):
                mutant = self.mutate(population, i)
                mutants[i] = np.clip(mutant, -1, 1)

            if self.cr_strategy=='bin':
                trials = self.crossover_bin(population, mutants)
            elif self.cr_strategy=='exp':
                trials = self.crossover_exp(population, mutants)
            else:
                trials = self.crossover_p(population, mutants, self.FE)
            trial_fitnesses = np.zeros(self.population_size)
            trial_rewards = []
            for i , trial in enumerate(trials):
                trial_fitness, trial_reward = problem.play_episode(trial)
                trial_fitnesses[i] = trial_fitness
                trial_rewards.append(trial_reward)
                self.FE += 1


            improved = self.selection(fitnesses, trial_fitnesses)

            population[improved] = trials[improved]
            fitnesses[improved] = trial_fitnesses[improved]
            for i, mark in enumerate(improved):
                if mark:
                    rewards[i] = trial_rewards[i]

            iteration += 1
            idx_prime = np.argmax(fitnesses)
            self.fitness_history.append(fitnesses[idx_prime])
            self.solution_history.append(population[idx_prime])
            self.reward_history.append(rewards[idx_prime])
            self.FE_history.append(self.FE)

        problem.show(population[idx_prime])

        return fitnesses[idx_prime], population[idx_prime], rewards[idx_prime]
