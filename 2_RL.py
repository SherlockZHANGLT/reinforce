import gym
import numpy as np
from matplotlib import pyplot as plt


class Arguments:
    def __init__(self):
        self.env = None
        self.obs_n = None
        self.act_n = None
        self.agent = None

        # Set your parameters here
        self.episodes =500
        self.max_step =1000
        self.lr =0.1
        self.gamma =0.9
        self.epsilon =0.1


class QLearningAgent:
    def __init__(self, args):
        self.obs_n = args.obs_n
        self.act_n = args.act_n
        self.lr = args.lr
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.Q = np.zeros((args.obs_n, args.act_n))

    def select_action(self, obs, if_train=True):
        if(if_train):
            if np.random.rand() <= self.epsilon:
                action= np.random.choice(self.act_n)
            else:
                action= self.argmax(obs)
        else:
            action=self.argmax(obs)
        return action

    def argmax(self,obs):
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]  # maxQ可能对应多个action
        action = np.random.choice(action_list)
        return action

    def update(self, transition):
        obs, action, reward, next_obs, done = transition
        predict_Q=self.Q[obs,action]
        if done:
            target_Q=reward
        else:
            target_Q=reward+self.gamma*np.max(self.Q[next_obs,:])
        self.Q[obs,action]+=self.lr*(target_Q-predict_Q)


class SARSAAgent:
    def __init__(self, args):
        self.obs_n = args.obs_n
        self.act_n = args.act_n
        self.lr = args.lr
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.Q = np.zeros((args.obs_n, args.act_n))

    def select_action(self, obs, if_train=True):
        if(if_train):
            if np.random.rand() <= self.epsilon:
                action= np.random.choice(self.act_n)
            else:
                action= self.argmax(obs)
        else:
            action=self.argmax(obs)
        return action

    def argmax(self,obs):
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]  # maxQ可能对应多个action
        action = np.random.choice(action_list)
        return action

    def update(self, transition):
        obs, action, reward, next_obs, next_action, done = transition
        predict_Q=self.Q[obs,action]
        if done:
            target_Q=reward
        else:
            target_Q=reward+self.gamma*self.Q[next_obs,next_action]
        self.Q[obs,action]+=self.lr*(target_Q-predict_Q)


def q_learning_train(args):
    env = args.env
    agent = args.agent
    episodes = args.episodes
    max_steps = args.max_step
    rewards = []
    mean_100ep_reward = []
    for episode in range(episodes):
        episode_reward = 0
        obs=env.reset()
        for t in range(max_steps):
            action=agent.select_action(obs)
            next_obs,reward,done,_=env.step(action)
            transition=obs, action, reward, next_obs, done
            agent.update(transition)
            obs=next_obs
            episode_reward += reward
            if done:
                 break
        print('Episode '+str(episode)+'\t Step '+str(t)+'\t Reward '+str(episode_reward))
        rewards.append(episode_reward)
        if len(rewards) < 100:
            mean_100ep_reward.append(np.mean(rewards))
        else:
            mean_100ep_reward.append(np.mean(rewards[-100:]))
    return mean_100ep_reward


def sarsa_train(args):
    env = args.env
    agent = args.agent
    episodes = args.episodes
    max_steps = args.max_step
    rewards = []
    mean_100ep_reward = []
    for episode in range(episodes):
        episode_reward = 0
        obs=env.reset()
        action=agent.select_action(obs)
        for t in range(max_steps):
            next_obs,reward,done,_=env.step(action)
            next_action=agent.select_action(next_obs)
            transition=obs, action, reward, next_obs, next_action, done
            agent.update(transition)
            action=next_action
            obs=next_obs
            episode_reward += reward
            if done:
                 break
        print('Episode '+str(episode)+'\t Step '+str(t)+'\t Reward '+str(episode_reward))
        rewards.append(episode_reward)
        if len(rewards) < 100:
            mean_100ep_reward.append(np.mean(rewards))
        else:
            mean_100ep_reward.append(np.mean(rewards[-100:]))
    return mean_100ep_reward


def q_learning_test(args):
    env = args.env
    agent = args.agent
    total_reward=0
    obs=env.reset()
    i=1
    print('q_learning_test: ')
    print(i)
    while True:
        i=i+1
        action=agent.select_action(obs,False)
        next_obs,reward,done,_=env.step(action)
        total_reward+=reward
        obs=next_obs
        env.render()
        if done:
            break
        print(i)
    return total_reward


def sarsa_test(args):
    env = args.env
    agent = args.agent
    total_reward=0
    obs=env.reset()
    i=1
    print('sarsa_test: ')
    print(i)
    while True:
        i=i+1
        action=agent.select_action(obs,False)
        next_obs,reward,done,_=env.step(action)
        total_reward+=reward
        obs=next_obs
        env.render()
        if done:
            break
        print(i)
    return total_reward



if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)

    q_learning_args = Arguments()
    env = gym.make("CliffWalking-v0")
    q_learning_args.env = env
    q_learning_args.obs_n = env.observation_space.n
    q_learning_args.act_n = env.action_space.n
    q_learning_args.agent = QLearningAgent(q_learning_args)

    sarsa_args = Arguments()
    env = gym.make("CliffWalking-v0")
    sarsa_args.env = env
    sarsa_args.obs_n = env.observation_space.n
    sarsa_args.act_n = env.action_space.n
    sarsa_args.agent = SARSAAgent(sarsa_args)

    q_learning_rewards = q_learning_train(q_learning_args)
    sarsa_rewards = sarsa_train(sarsa_args)

    q_learning_test_rewards=q_learning_test(q_learning_args)
    sarsa_test_rewards=sarsa_test(sarsa_args)

    print('q_learning_test_reward = %.1f'%(q_learning_test_rewards))
    print('sarsa_test_rewards = %.1f'%(sarsa_test_rewards))

    plt.plot(range(q_learning_args.episodes), q_learning_rewards, label='Q Learning')
    plt.plot(range(sarsa_args.episodes), sarsa_rewards, label='SARSA')
    plt.legend()
    plt.show()