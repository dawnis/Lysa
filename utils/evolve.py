import evo_rl
from .environment_classes import *

class evolution_chamber:

    def __init__(self, environment):

        self.configuration = {
            "population_size": 200,
            "survival_rate": 0.2,
            "mutation_rate": 0.4,
            "input_size": 8,
            "output_size": 1,
            "topology_mutation_rate": 0.4,
            "project_name": "Lysa",
            "project_directory": "/Users/dchow/git/Lysa/agents"
        }

        self.population = evo_rl.PopulationApi(self.configuration)

        self.max_generations = 1000
        self.stopping_fitness = 400

        if environment == 'CartPole':
            self.env = CartPoleEnvironment(self.configuration)
        elif environment == 'MountainCar':
            self.env = MountainCarEnvironment(self.configuration)
        else:
            self.env = None


    def run_evolve(self):

        while self.population.generation < self.max_generations:
            for agent in range(self.configuration["population_size"]):
                self.env.evaluate_agent(self.population, agent)
            if self.population.fitness > self.stopping_fitness:
                break

            self.population.update_population_fitness()
            self.population.report()
            self.population.evolve_step()
