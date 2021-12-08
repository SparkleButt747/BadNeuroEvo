import time

import Box2D
import Box2D.b2

import gym
from gym import spaces
from gym.utils import colorize, seeding, EzPickle
import numpy as np

env = gym.make('Box2D:BipedalWalkerHardcore-v3')
observations = env.reset()

reward = 0


import neural_network_NE_agent_2
neural_network_NE_agent_2.Evolution.simulate_generation(self=neural_network_NE_agent_2.Evolution, observation=None, is_dead=True, score=0, first_time=True)
first_time = False

done = False


for i in range(10000):
    for q in range(neural_network_NE_agent_2.genomes):
        start_timer = time.time()
        while not done:
            output_signal = neural_network_NE_agent_2.Evolution.simulate_generation(self=neural_network_NE_agent_2.Evolution, observation=observations, is_dead=False, score=reward, first_time=False)
            observations, reward, done, info = env.step(output_signal[0])


        done = False
        observations = env.reset()
        neural_network_NE_agent_2.Evolution.simulate_generation(self=neural_network_NE_agent_2.Evolution, observation=None, is_dead=True, score=0, first_time=False)
