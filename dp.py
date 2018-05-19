# this is a demo for dynamic programming of Reinforcement Learning
import numpy as np
from pandas import DataFrame
from simple_env import SimpleEnv


class Dp:
    def __init__(self, env, reward_discount=0.9):
        self.actions = env.actions_space()
        self.gamma = reward_discount
        self.env = env
        self.total_states = env.total_states()
        action_p = np.ones(env.total_states())
        self.policy = DataFrame(data=list(zip(action_p, action_p, action_p, action_p)),
                                columns=['up', 'down', 'left', 'right'])
        self.state_value = np.zeros(env.total_states())

    def choose_action(self):
        pass

    def learn(self):
        pass

    def print(self):
        print(self.policy)
        print(self.state_value.reshape(self.env.size()))

    def compute_state_value(self, state):
        sum_value = 0.0
        sum_rate = 0.0
        for action in self.policy.columns:
            next_state, reward = self.env.step_with_state(action, state)
            sum_value += self.policy[action][state] * self.state_value[next_state] * self.gamma + reward
            sum_rate += self.policy[action][state]
        if sum_rate == 0.0:
            return 0.0
        return sum_value / sum_rate

    def policy_evaluation(self):
        new_state_value = np.zeros(self.total_states)
        for i in range(1, self.total_states - 1):
            new_state_value[i] = self.compute_state_value(i)
        self.state_value = new_state_value

    def single_state_policy(self, state):
        values = np.array([float('-inf') for _ in range(0, len(self.actions))])
        new_policy = np.zeros(len(self.actions))
        for action in self.policy.columns:
            action_index = self.policy.columns.get_loc(action)
            next_state, _ = self.env.step_with_state(action, state)
            if self.policy[action][state] > 0.0 and state != next_state:
                values[action_index] = self.state_value[next_state]
        max_value = values.max()
        for i in range(0, len(self.actions)):
            if values[i] == max_value:
                new_policy[i] = 1.0
            else:
                new_policy[i] = 0.0
        return new_policy

    def policy_iteration(self):
        for s in range(1, self.total_states - 1):
            self.policy.loc[s] = self.single_state_policy(s)

if __name__ == '__main__':
    env = SimpleEnv(4, 4)
    dp = Dp(env, 1)
    for i in range(200):
        env.reset()
        dp.policy_evaluation()
        dp.policy_iteration()
        dp.print()
