import gym
import numpy as np
import math


class RL:
    def __init__(self, learning_rate, greedy, discount, n_actions, delay_rate=0.99, delay_steps=1000):
        self.epsilon = greedy
        self.delay_rate = delay_rate
        self.delay_steps = delay_steps
        self.lr = learning_rate
        self.discount = discount
        self.n_actions = n_actions
        self.global_steps = 0

    def choose_action(self, o):
        pass

    def learning_rate(self):
        return self.lr * math.pow(self.delay_rate, self.global_steps / self.delay_steps)


# polynomial linear model for semi-gradient TD(0)
class SemiGradientTD0(RL):
    def __init__(self, n_actions=0, learning_rate=0.1, greedy=0.9, discount=0.9):
        RL.__init__(self, learning_rate, greedy, discount, n_actions)
        # 9 features and 1 action
        self.weight = np.ones([10, 1])

    def q_prediction(self, observation, action):
        features = self.obs_action_to_features(observation, action)
        return np.matmul(features, self.weight)[0][0]

    @staticmethod
    def obs_action_to_features(observation, action):
        # two dimensions of observation
        s0 = observation[0]
        s1 = observation[1]
        features = [1, s0, s1, s0 * s1, s0 * s0, s1 * s1, s0 * s1 * s1, s1 * s0 * s0, s0 * s0 * s1 * s1, action]
        return np.array(features).reshape(1, 10)

    def learn(self, observation, action, reward, _observation, _action, done):
        self.global_steps += 1
        if done:
            q_target = reward
        else:
            q_target = reward + self.discount * self.q_prediction(_observation, _action)
        q_pre = self.q_prediction(observation, action)
        w_err = self.learning_rate() * (q_target - q_pre) * self.obs_action_to_features(observation, action).reshape(10, 1)
        self.weight = self.weight + w_err

    # epsilon-greedy
    def choose_action(self, observation):
        if np.random.rand() < self.epsilon:
            q_max = float('-inf')
            max_index = -1
            for i in range(self.n_actions):
                q = self.q_prediction(observation, i)
                if q_max < q:
                    q_max = q
                    max_index = i
            action = max_index
        else:
            action = np.random.randint(0, self.n_actions)
        return action


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    brain = SemiGradientTD0(env.action_space.n)
    for i_episode in range(20000):
        observation = env.reset()
        action = brain.choose_action(observation)
        t = 0
        while True:
            t += 1
            env.render()
            _observation, reward, done, info = env.step(action)
            _action = brain.choose_action(_observation)
            position, velocity = _observation
            reward = abs(position - (-0.5))
            brain.learn(observation, action, reward, _observation, _action, done)
            observation = _observation
            action = _action
            if done:
                print(brain.weight, brain.learning_rate())
                print("Episode {} finished after {} times steps".format(i_episode, t))
                break
