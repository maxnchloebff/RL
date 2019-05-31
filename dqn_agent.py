import numpy as np
import tensorflow as tf
import time
import sys
import matplotlib.pyplot as plt

class DQN_agent:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,  # 每走300步copy一次
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=None,  # e_greedy增量
                 output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self._build_net()
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t,e) for t,e in zip(t_params,e_params)]  # 通过e来更新t
        self.sess = tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/",self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def build_net(self):
        self.s = tf.placeholder(tf.float32,[None,self.n_features],name='s')  # 状态
        self.s_ = tf.placeholder(tf.float32,[None,self.n_features],name='s_')  # 下一个状态
        self.r = tf.placeholder(tf.float32,[None,],name='r')  # 奖励
        self.a = tf.placeholder(tf.int32,[None,],name='a')  # 行为
        w_initializer,b_initializer = tf.random_normal_initializer(0.,0.3),tf.constant_initializer(0.1)  # 初始化参数

        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s,20,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='e1')  # 参数分别为输入数据张量状态，神经单元数，激活函数，全连接层参数
            self.q_eval = tf.layers.dense(e1,self.n_actions,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='q')  # 再来一层全连接层，将上一层输出输入进去，输出单元数为n_actions

        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_,20,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='t1')  # 下一个状态，中间单元数20，两个网络结构相同
            self.q_next = tf.layers.dense(t1,self.n_actions,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='t2')  # 应该是每一个action对应一个reward

        with tf.variable_scope('q_target'):  # 用target网络生成的q_target
            # 利用target_net生成的结果， 得到最大的Q
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next,axis=1,name='Qmax_s_')
            # 梯度下降法
            self.q_target = tf.stop_gradient(q_target)  # 如何理解stop_gradient

        with tf.variable_scope('a_eval'):  # 用来计算loss
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0],dtype=tf.int32),self.a],axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval,indices=a_indices)

        with tf.variable_scope('loss'):  # 损失函数
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval_wrt_a,name='TD_error'))

        with tf.variable_scope('train'):  # 训练网络
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)  # 根据学习率来优化使得loss最小化

    def store_transition(self,s,a,r,s_):
        if not hasattr(self,'memory_counter'):  # 判断对象是否包含对应的属性
            self.memory_counter = 0
        transition = np.hstack((s,[a,r],s_))  # 水平穿在一起
        index = self.memory_counter % self.memory_size  # 计算存储的行索引
        self.memory[index,:] = transition
        self.memory_counter += 1

    def choose_action(self,observation):
        observation = observation[np.newaxis,:]
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval,feed_dict={self.s:observation})
            action = np.argmax(actions_value)  # 选择令Q网络值最大的action
        else:
            action = np.random.randint(0,self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)  # 执行eval_net赋值给target_net的复制网络，主要是复制参数
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:  # 随机采样一组batch_size大小的样本
            sample_index = np.random.choice(self.memory_size,size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter,size=self.batch_size)
        batch_memory = self.memory[sample_index,:]

        _,cost = self.sess.run(
            [self._train_op,self.loss],
            feed_dict={
                self.s:batch_memory[:,:self.n_features],  # 前n_features是state
                self.a:batch_memory[:,self.n_features],  # 第n_features是aaction
                self.r:batch_memory[:,self.n_features + 1],  # reward
                self.s_:batch_memory[:,-self.n_features:],  # 下一个state
            }
        )

        self.cost_his.append(cost)  # 损失值

        # 动态改变epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1  # 学习数

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)),self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
