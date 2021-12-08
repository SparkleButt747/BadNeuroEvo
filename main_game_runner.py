import gym
from gym import envs

print(envs.registry.all())
import random


env = gym.make('LunarLander-v2')
observation = env.reset()
print(observation)
print(env.action_space)
print()
reward = 0

import neural_network_NE_agent
neural_network_NE_agent.Evolution.simulate_generation(self=neural_network_NE_agent.Evolution, observation=None, is_dead=True, score=0, first_time=True)
first_time = False

done = False

for i in range(50000):
    print('Generation: ' + str(i))
    for q in range(neural_network_NE_agent.genomes):
        while not done:
            env.render()
            output_signal = neural_network_NE_agent.Evolution.simulate_generation(self=neural_network_NE_agent.Evolution, observation=[observation], is_dead=False, score=reward, first_time=False)
            observation, reward, done, info = env.step(output_signal)

        done = False
        observation = env.reset()
        neural_network_NE_agent.Evolution.simulate_generation(self=neural_network_NE_agent.Evolution, observation=None,is_dead=True, score=0, first_time=False)
