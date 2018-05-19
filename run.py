from maze_env import Maze
from RL_brain_table import SarsaTable, SarsaLambdaTable, QLearningTable


def q_update():
    for episode in range(100):
        observation = env.reset()
        step = 0
        while True:
            step += 1
            env.render()
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            RL.learn(str(observation), action, reward, str(observation_))
            observation = observation_
            if done:
                break
        # cost_his.append(step)
    print('game over')
    env.destroy()


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        if hasattr(RL, 'eligible_trace'):
            RL.eligible_trace *= 0

        # RL choose action based on observation
        action = RL.choose_action(str(observation))

        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation and action
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    aiType = input("请选择算法:\n1.SarsaTable\n2.SarsaLambdaTable\n3.Q_learning\n")
    env = Maze()
    if '1' == aiType:
        print("using sarsaTable algorithm")
        RL = SarsaTable(actions=list(range(env.n_actions)))
        env.after(100, update)
    elif '2' == aiType:
        print("using sarsaLambdaTable algorithm")
        lambdaVal = input("请输入lambda:\n")
        RL = SarsaLambdaTable(actions=list(range(env.n_actions)), lambda_value=float(lambdaVal))
        env.after(100, update)
    else:
        print("using Q_learning algorithm")
        RL = QLearningTable(actions=list(range(env.n_actions)))
        env.after(100, q_update)
    env.mainloop()
