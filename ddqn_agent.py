import tensorflow as tf
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


class DDQN_Agent:

    def __init__(self, ob, dim_action, reward_decay, l_rate, bacth_size, memory_size, epsilon,
                 output_graph):
        self.ob = ob
        self.dim_action = dim_action
        self.gamma = reward_decay
        self.memory_size = memory_size
        self.l_rate = l_rate
        self.batch_size = bacth_size
        self.episode = 0
        self.epsilon = epsilon

        self.memory = np.zeros((memory_size, ob * 2 + 2))

        print("ob", ob)
        print("dim_action", dim_action)

        self.trainsition_count = 0

        num_steps_sum = 0

        self.target_params = tf.get_collection("target_network_para")
        self.evaluate_params = tf.get_collection("evaluate_network_para")

        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        self.cost_list = []

        self.episode_count = 0

        self.epsilon_base = epsilon

        self.build_network()

        self.sess.run(tf.global_variables_initializer())

        self.step_counter = 0

    def replace_target(self):
        # 把target_work 参数 赋值给 evaluate_network
        for target, evaluate in zip(self.target_params, self.evaluate_params):
            tf.assign(target, evaluate)

    def build_network(self):

        #define input
        self.state = tf.placeholder(tf.float32, [None, self.ob], name="input_state")
        self.q_target = tf.placeholder(tf.float32, [None, self.dim_action], name="input_q_target")

        # define network and parameters

        with tf.variable_scope("evaluate_network"):
            c_names, n_l1, w_initializer, b_initializer = ["evaluate_network_para", tf.GraphKeys.GLOBAL_VARIABLES], \
                                                          10, \
                                                          tf.random_normal_initializer(0.0, 0.3), \
                                                          tf.constant_initializer(0.2)

            with tf.variable_scope("l1"):
                w1 = tf.get_variable("w1", [self.ob, n_l1], initializer=w_initializer, collections=c_names)
                # w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable("b1", [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.state, w1) + b1)

            with tf.variable_scope("l2"):
                w2 = tf.get_variable("w2", [n_l1, self.dim_action], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable("b2", [1, self.dim_action], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_eval, self.q_target))

        with tf.variable_scope("train"):
            self.train_op = tf.train.RMSPropOptimizer(self.l_rate).minimize(self.loss)


        self.state_next = tf.placeholder(tf.float32, [None, self.ob], name="input_next_state")
        with tf.variable_scope("target_network"):
            c_names, n_l1, w_initializer, b_initializer = ["target_network_para",
                                                           tf.GraphKeys.GLOBAL_VARIABLES], 10, tf.random_normal_initializer(
                0., 0.3), tf.constant_initializer(0.2)

            with tf.variable_scope("l1"):
                w1 = tf.get_variable("w1", [self.ob, n_l1], initializer=w_initializer,
                                     collections=c_names)
                b1 = tf.get_variable("b1", [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.state_next, w1) + b1)

            with tf.variable_scope("l2"):
                w2 = tf.get_variable("w1", [n_l1, self.dim_action], initializer=w_initializer,
                                     collections=c_names)
                b2 = tf.get_variable("b2", [1, self.dim_action], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_trainsition(self, s, a, r, s_):
        item = np.hstack((s, [a, r], s_))
        index = self.trainsition_count % self.memory_size
        self.memory[index, :] = item
        self.trainsition_count = self.trainsition_count + 1

    def choose_action(self, observation):
        random_value = random.random()
        observation = observation[np.newaxis, :]
        if random_value < self.epsilon:  # the possibility
            action_values = self.sess.run(self.q_eval, feed_dict={self.state: observation})
            #    action_values = self.sess.run(self.q_eval,feed_dict={self.s    :observation})

            action = np.argmax(action_values)
        else:
            action = np.random.randint(0, self.dim_action)
        return action

    def learning(self):

        if self.episode > 90:
            self.epsilon = 1 - (1 - self.epsilon_base) * (400 / (400 + self.episode))
        else:
            self.epsilon = self.epsilon_base

        # print("current transition amount：%s"%self.trainsition_count)

        if self.trainsition_count < self.batch_size:
            print("continue")

        if self.step_counter % 50 == 0:
            self.replace_target()

        if self.batch_size <= self.trainsition_count and self.trainsition_count <= self.memory_size:
            # print("self.trainsition_count",self.trainsition_count)
            sample_index = np.random.choice(self.trainsition_count, size=self.batch_size)
        else:
            # print("self.memory_size",self.memory_size)
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)

        batch = self.memory[sample_index, :]

        #  bacth中每一行的格式为： s,a,r,s_
        #  学习过程：根据batch，拆成 s a r s_
        #          (1)  s 结合 实时更新的evaluate_network 生成 q_evaluate
        #         （2）  根据r ,以及，s_结合迟滞更新的target_network得到max的q_next，得到q_target = r + gamma*q_next

        s = batch[:, :self.ob]
        r = batch[:, self.ob + 1]
        s_ = batch[:, -self.ob:]
        a = batch[:, self.ob].astype(int)

        q_next, q_evaluate = self.sess.run([self.q_next, self.q_eval], feed_dict={self.state_next: s_, self.state: s})

        # q_target，using q_target to calculate self.loss ,rewrite q_target = r + gamma*q_next
        q_target = q_evaluate.copy()

        q_target[:, a] = r + self.gamma * np.max(q_next, axis=1)

        _, cost = self.sess.run([self.train_op, self.loss],
                                feed_dict={self.q_target: q_target, self.state: s})  # 只针对evaluate_network训练

        # print("第 %s 轮的cost：%s ,当前epsilon大小为%s"%(self.episode,cost,self.epsilon))

        self.cost_list.append(cost)

    def plot_cost(self):
        
        plt.plot(np.arange(len(self.cost_list)), self.cost_list)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
