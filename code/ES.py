'''
Requirements: 
    pip install "gymnasium[box2d]" numpy matplotlib
'''
from problem import GymProblem
import numpy as np
import matplotlib.pyplot as plt

class EvolutionStrategy:

    def __init__(self, sigma, max_iterations):
        self.sigma = sigma
        self.max_iterations = max_iterations
        self.fitness_history = []
        self.reward_history = []
        self.FE = 0
        self.FE_history = []

    def mutate(self, parent, sigma):
        offspring = parent + np.random.normal(0, sigma, parent.shape)
        return np.clip(offspring, -1, 1)

    def __call__(self, problem):
        parent = problem.sample()
        fitness_prime, best_rewards = problem.play_episode(parent)
        self.FE += 1

        for i in range(self.max_iterations):
            offspring = self.mutate(parent, self.sigma)
            fitness, rewards = problem.play_episode(offspring)
            self.FE += 1

            if fitness > fitness_prime:
                fitness_prime = fitness
                parent = offspring
                best_rewards = rewards

            self.fitness_history.append(fitness_prime)
            self.reward_history.append(rewards)
            self.FE_history.append(self.FE)

        problem.show(parent)
        return fitness_prime




