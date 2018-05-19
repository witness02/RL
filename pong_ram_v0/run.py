import gym
from pong_ram_v0.RL_brain import DQNPrioritizedReplay
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

env = gym.make('Pong-ram-v0')
env = env.unwrapped
env.seed(21)
MEMORY_SIZE = 10000

sess = tf.Session()
with tf.variable_scope('DQN_with_prioritized_replay'):
    RL_prio = DQNPrioritizedReplay(
        n_actions=4, n_features=128, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=True, output_graph=True,
    )
sess.run(tf.global_variables_initializer())


def train(RL):
    total_steps = 0
    steps = []
    episodes = []
    # for i_episode in range(20000000000):
    i_episode = 0
    while True:
        i_episode += 1
        observation = env.reset()
        e_rewards = 0
        while True:
            env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            if done: reward = 10
            e_rewards += reward
            RL.store_transition(observation, action, reward, observation_)

            if total_steps > MEMORY_SIZE:
                RL.learn()

            if done:
                print('episode ', i_episode, ' finished, total reward is ', e_rewards)
                steps.append(total_steps)
                episodes.append(i_episode)
                break

            observation = observation_
            total_steps += 1
    return np.vstack((episodes, steps))

his_prio = train(RL_prio)

