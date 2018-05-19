import gym
import random

env = gym.make('Maze-v0')
# env = gym.make('GridWorld-v0')
env.reset()
print(env.action_space)
for _ in range(1000):
	env.render()
	env.step(env.action_space[random.randint(0, len(env.action_space)-1)])
