import tensorflow as tf
import numpy as np
import gym


class DQNBrain:
    def __init__(self,
                 n_features=2,
                 n_actions=2,
                 memory_size=500,
                 e_greedy=0.9,
                 e_greedy_incr=None,
                 reward_decay=0.9,
                 learning_rate=0.01,
                 replace_target_iter=300,
                 batch_size=32
                 ):
        self.memory_size = memory_size
        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_incr
        self.epsilon = 0 if e_greedy_incr is not None else self.epsilon_max
        self.gamma = reward_decay
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = learning_rate
        self.memory_counter = 0
        self.learn_step_counter = 0
        self.replace_target_iter = replace_target_iter
        self.batch_size = batch_size

        # init memory [s, a, r, s_]
        try:
            self.memory = np.load('./memory.npy')
            print(self.memory)
        except:
            self.memory = np.zeros((memory_size, 2 * n_features + 2))

        self.cost_hist = []

        tf.reset_default_graph()
        self.build_network()

        # define replacing operator
        t_params = tf.get_collection('target_network_params')
        e_params = tf.get_collection('evaluate_network_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        try:
            self.saver.restore(self.sess, "./model.ckpt")
            print("restore weight")
        except:
            pass

    def build_network(self):
        # build evaluate network
        self.s = tf.placeholder(tf.float32, (None, self.n_features), name='s')
        self.q_target = tf.placeholder(tf.float32, (None, self.n_actions), name='q_target')
        with tf.variable_scope("evaluate_network"):
            c_name = ["evaluate_network_params", tf.GraphKeys.GLOBAL_VARIABLES]
            n_l1 = 10
            w_initializer = tf.random_normal_initializer(0., 0.3)
            b_initializer = tf.constant_initializer(0.1)
            with tf.variable_scope("l1"):
                w1 = tf.get_variable("w1", initializer=w_initializer, shape=[self.n_features, n_l1], collections=c_name)
                b1 = tf.get_variable("b1", initializer=b_initializer, shape=[1, n_l1], collections=c_name)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
            with tf.variable_scope("l2"):
                w2 = tf.get_variable("w2", initializer=w_initializer, shape=[n_l1, self.n_actions], collections=c_name)
                b2 = tf.get_variable("b2", initializer=b_initializer, shape=[1, self.n_actions], collections=c_name)
                self.q_eval = tf.matmul(l1, w2) + b2
            with tf.variable_scope("loss"):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            with tf.variable_scope("train"):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # build target network for storing weight
        self.s_ = tf.placeholder(tf.float32, (None, self.n_features), name="s_")
        with tf.variable_scope("target_network"):
            c_name = ["target_network_params", tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope("l1"):
                w1 = tf.get_variable("w1", initializer=w_initializer, shape=[self.n_features, n_l1], collections=c_name)
                b1 = tf.get_variable("b1", initializer=b_initializer, shape=[1, n_l1], collections=c_name)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
            with tf.variable_scope("l2"):
                w2 = tf.get_variable("w2", initializer=w_initializer, shape=[n_l1, self.n_actions], collections=c_name)
                b2 = tf.get_variable("b2", initializer=b_initializer, shape=[1, self.n_actions],collections=c_name)
                self.q_next = tf.matmul(l1, w2) + b2

    # epsilon-greedy
    def choose_action(self, s):
        s = s[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            action_value = self.sess.run(self.q_eval, feed_dict={self.s: s})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        # equal to self.memory[index, :] = transition
        self.memory[index] = transition
        self.memory_counter += 1

    def learn(self):
        self.learn_step_counter += 1
        if self.learn_step_counter % self.replace_target_iter == 0:
            print("replace target network")
            self.sess.run(self.replace_target_op)
            np.save('./memory', self.memory)
            self.saver.save(self.sess, "./model.ckpt")

        # sample
        if self.memory_counter < self.memory_size:
            sample_index = np.random.choice(self.memory_counter, self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size, self.batch_size)

        sample = self.memory[sample_index, :]
        q_next, q_eval = self.sess.run([self.q_next, self.q_eval],
                                       feed_dict={
                                           self.s:sample[:, :self.n_features],
                                           self.s_: sample[:, -self.n_features:]
                                       })
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        actual_eval_index = sample[:, self.n_features].astype(int)
        reward = sample[:, self.n_features + 1]

        q_target[batch_index, actual_eval_index] = reward + self.gamma * np.max(q_next, axis=1)
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: sample[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_hist.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    brain = DQNBrain(n_actions=3, n_features=2, learning_rate=0.01, e_greedy=0.9, memory_size=3000, e_greedy_incr=0.0002)
    total_step = 0

    for i_episode in range(2000):
        observation = env.reset()
        step = 0
        episode_total_reward = 0
        while True:
            env.render()
            step += 1
            total_step += 1
            action = brain.choose_action(observation)
            observation_, reward, done, _ = env.step(action)
            position, velocity = observation_
            reward = abs(position - (-0.5))
            brain.store_transition(observation, action, reward, observation_)
            # if total_step > 1000:
            brain.learn()
            observation = observation_
            episode_total_reward += reward
            if done:
                print("eposide %d use %d steps total reward is %d" % (i_episode, step, episode_total_reward))
                break
