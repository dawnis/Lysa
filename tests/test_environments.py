import unittest
from utils.environment_classes import *

class TestCCEnvironment(unittest.TestCase):

    def test_observation_mapper(self):
        obs_space_dictionary = {
            "xpos": 0,
            "xvel": 1,
            "angle": 2
        }

        observation = [0.2, 5, 42]
        vector = map_observation_8d(observation, obs_space_dictionary)
        self.assertEqual(vector, [0.2, 0, 5, 0, 0, 42, 0, 0])


if __name__ == '__main__':
    unittest.main()