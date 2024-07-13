import argparse
import sys, os
from evo_rl import AgentApi

parent_directory = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.append(parent_directory)

from utils.environment_classes import *

class Observer:

    def __init__(self, environment, checkpoint):

        self.configuration = {
            "synaptic_mutation_rate": 0,
            "topology_mutation_rate": 0,
            "mutation_effect": 0,
        }
        self.env = get_gym_env(environment, self.configuration)
        self.checkpt = checkpoint
        self.agentApi = AgentApi(self.configuration, checkpoint)


    def view(self):
        self.env.observe(self.agentApi)


def observer_script():
    parser = argparse.ArgumentParser(description="Script to run observation of a checkpointed Agent in Pygame")
    parser.add_argument("--chkpt", type=str, help="checkpoint file")
    parser.add_argument("--env", type=str, help="environment to run")
    args = parser.parse_args()
    observer = Observer(args.env, args.chkpt)
    observer.view()


if __name__ == "__main__":
    observer_script()


