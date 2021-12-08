import time
import numpy as np
import Box2D
import Box2D.b2

import gym
import gym3
from gym import spaces
from gym.utils import colorize, seeding, EzPickle
import numpy as np

env = gym.make('procgen:procgen-chaser-v0')
env.render(mode='human')
observation = env.reset()
reward = 0

new_observation = np.reshape(observation, newshape=[64*64*3])

import neural_network_NE_agent_3
neural_network_NE_agent_3.Evolution.simulate_generation(self=neural_network_NE_agent_3.Evolution, observation=None, is_dead=True, score=0, first_time=True)
first_time = False

done = False


for i in range(10000):
    for q in range(neural_network_NE_agent_3.genomes):
        while not done:
            env.render()
            output_signal = neural_network_NE_agent_3.Evolution.simulate_generation(self=neural_network_NE_agent_3.Evolution, observation=new_observation, is_dead=False, score=reward, first_time=False)
            observations, reward, done, info = env.step(output_signal)
            new_observation = np.reshape(observations, newshape=[64*64*3])

        done = False
        observations = env.reset()
        neural_network_NE_agent_3.Evolution.simulate_generation(self=neural_network_NE_agent_3.Evolution, observation=None, is_dead=True, score=0, first_time=False)
