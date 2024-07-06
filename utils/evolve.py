import evo_rl
from .environment_classes import *

class evolution_chamber:

    def __init__(self, environment, checkpoint=None):

        self.configuration = {
            "population_size": 2000,
            "survival_rate": 0.05,
            "mutation_rate": 0.4,
            "input_size": 8,
            "hidden_size": 2,
            "output_size": 1,
            "topology_mutation_rate": 0.4,
            "project_name": "Lysa",
            "project_directory": "/Users/dchow/git/Lysa/agents"
        }

        if checkpoint is not None:
            self.population = evo_rl.PopulationApi(self.configuration, checkpoint)
        else:
            self.population = evo_rl.PopulationApi(self.configuration)

        self.max_generations = 200
        self.min_generations = 10 #so we can reach the visualization checkpoint
        self.stopping_fitness = 200

        if environment == 'CartPole':
            self.env = CartPoleEnvironment(self.configuration)
        elif environment == 'MountainCar':
            self.env = MountainCarEnvironment(self.configuration)
        elif environment == 'Acrobot':
            self.env = AcrobotEnvironment(self.configuration)
        elif environment == 'Pendulum':
            self.env = PendulumEnvironment(self.configuration)
        else:
            self.env = None


    def run_evolve(self):

        while self.population.generation < self.max_generations:

            for agent in range(self.configuration["population_size"]):
                self.env.evaluate_agent(self.population, agent)

            if (self.population.generation > self.min_generations) and (self.population.fitness > self.stopping_fitness):
                break

            self.population.update_population_fitness()
            self.population.report()
            self.population.evolve_step()
