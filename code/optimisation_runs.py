from problem import GymProblem
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ES import EvolutionStrategy
from DE import DifferentialEvolution




# Function to run optimisation runs for the ES algorithm
def run_optimisation_es(runs):
  all_fitness_histories = []
  all_reward_histories = []
  all_FE_histories = []

  for _ in range(runs):
    problem = GymProblem("LunarLander-v3", continuous=False, gravity=-10, enable_wind=True, wind_power=0, turbulence_power=0) 
    test_run = EvolutionStrategy(sigma=0.4, max_iterations=10)
    best = test_run(problem)
    all_fitness_histories.append(test_run.fitness_history)
    all_reward_histories.append(test_run.reward_history)
    all_FE_histories.append(test_run.FE_history)


  return all_fitness_histories, all_reward_histories, all_FE_histories


# Function to run optimisation runs for the DE algorithms
def run_optimisation_de(runs, strategy=None):
  all_fitness_histories = []
  all_solution_histories = []
  all_reward_histories = []
  final_fitnesses = []
  all_FE_histories = []

  for run in range(runs):
    problem = GymProblem("LunarLander-v3", continuous=False, gravity=-10, enable_wind=True, wind_power=0, turbulence_power=0)
    test_run = DifferentialEvolution(population_size=20, F=0.8, cr=0.9, max_iterations=1000, cr_strategy=strategy)
    best, sol, reward = test_run(problem)
    all_fitness_histories.append(test_run.fitness_history)
    all_solution_histories.append(test_run.solution_history)
    all_reward_histories.append(test_run.reward_history)
    final_fitnesses.append(best)
    all_FE_histories.append(test_run.FE_history)

  return all_fitness_histories, all_solution_histories, all_reward_histories, final_fitnesses, all_FE_histories

###############################################################################################################################################

# The plots for single algorithms
def plot_de_results(fitness_histories, reward_histories, FE_evals):
  
  # First plot is the best-so-far fitness on the y-axis and the amount of fitness evaluations performed on the x-axis
  runs_to_plot = len(fitness_histories)
  plt.figure(figsize=(10, 4))

  for run_id in range(runs_to_plot):
      plt.plot(FE_evals[run_id], fitness_histories[run_id],alpha=0.5)
  plt.xlabel("Fitness evaluation")
  plt.ylabel("Fitness")
  plt.grid(True)
  plt.show()

  # Second plot is rewards per simulation averaged over all runs
  max_steps = max(len(r) for run in reward_histories for r in run)
  avg_rewards = np.zeros(max_steps)
  count = np.zeros(max_steps)

  for run in reward_histories:
      for rewards in run:
          for step, reward in enumerate(rewards):
              avg_rewards[step] += reward
              count[step] += 1

  avg_rewards /= count

  plt.figure(figsize=(10, 4))
  plt.plot(avg_rewards, color='blue')
  plt.xlabel("Simulation Step")
  plt.ylabel("Average Reward")
  plt.grid(True)
  plt.show()


############################################################################################################################################

# Function automatically creating the csv containing all the desired values of the algorithms runs to plot the IOH analyzer
def save_results_to_csv(fitness_histories, FE_histories, algorithm_id, problem_id, dimension, filename, FE_max=1000):
    all_data = []
    for run_id, (fitness_history, fe_history) in enumerate(zip(fitness_histories, FE_histories)):
        for fval, fe in zip(fitness_history, fe_history):
            if fe > FE_max:
                  continue
            all_data.append({"Evaluation_counter": fe,
                              "Function_value": fval,
                              "Function_ID": problem_id,
                              "Algorithm_ID": algorithm_id,
                              "Problem_dimension": dimension,
                              "Run_ID": run_id})

    df = pd.DataFrame(all_data)
    df.to_csv(filename, index=False)

