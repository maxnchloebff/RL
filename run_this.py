import gym
from ddqn_agent import DDQN_Agent
curent_env = gym.make("CartPole-v0")

curent_env = curent_env.unwrapped
"""debug information"""

# print("observation:", my_env.observation_space)
# print("observation_max", my_env.observation_space.high)
# print("observation_mim", my_env.observation_space.low)
# print("actions:", my_env.action_space)
""""""

agent = DDQN_Agent(dim_observation=curent_env.observation_space.shape[0],
                            dim_action=curent_env.action_space.n,
                            reward_decay=0.95,
                            l_rate=0.05,
                            bacth_size=32,
                            memory_size=800,
                            epsilon=0.7,
                            output_graph=True)

total_steps = 0

for i in range(1000):

    i = agent.episode

    observation = curent_env.reset()

    epi_reward = 0

    while True:

        curent_env.render()

        action = agent.choose_action(observation)

        observation_, reward, done, info = curent_env.step(action)

        # construct reward
        x, x_dot, theta, theta_dot = observation_
        r1 = (curent_env.x_threshold - abs(x)) / curent_env.x_threshold - 0.8
        r2 = (curent_env.theta_threshold_radians - abs(theta)) / curent_env.theta_threshold_radians - 0.5
        reward = 0.6 * r1 + 0.4 * r2

        epi_reward = epi_reward + reward

        agent.store_trainsition(observation, action, reward, observation_)

        total_steps = total_steps + 1

        if total_steps > 1000:
            agent.learning()

        if done:
            break

        observation = observation_

    agent.episode = agent.episode + 1

    print("current %s th episode，total reward is %s，current epsilon：%s" % (i, epi_reward, agent.epsilon))

agent.plot_cost()


