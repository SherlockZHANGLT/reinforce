import copy
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
import time


class EnvMoveBox:
    def __init__(self):
        self.original_field = [[1, 1, 5, 5, 5, 1, 1],
                               [1, 0, 0, 0, 0, 0, 1],
                               [1, 0, 0, 0, 0, 0, 1],
                               [1, 0, 0, 0, 0, 0, 1],
                               [1, 0, 0, 4, 0, 0, 1],
                               [1, 2, 0, 0, 0, 3, 1],
                               [1, 1, 1, 1, 1, 1, 1]]
        self.action_map = {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1)}

    def reset(self):
        self.field = copy.deepcopy(self.original_field)
        self.a_pos = (5, 1)
        self.b_pos = (5, 5)
        self.box_pos = (4, 3)
        self.goal_pos = (0, 3)

    def move(self, pos, next_pos):
        if 0 <= next_pos[0] <= 6 and 0 <= next_pos[1] <= 6 and \
           self.field[next_pos[0]][next_pos[1]] not in [1, 2, 3, 4]:
            self.field[pos[0]][pos[1]], self.field[next_pos[0]][next_pos[1]] = \
                self.field[next_pos[0]][next_pos[1]], self.field[pos[0]][pos[1]]
            return next_pos
        else:
            return pos

    def step(self, action_list):
        a_box = abs(self.a_pos[0] - self.box_pos[0]) + abs(self.a_pos[1] - (self.box_pos[1] - 1))
        b_box = abs(self.b_pos[0] - self.box_pos[0]) + abs(self.b_pos[1] - (self.box_pos[1] + 1))
        box_goal = abs(self.box_pos[0] - self.goal_pos[0]) + abs(self.box_pos[1] - self.goal_pos[1])

        if self.a_pos == (self.box_pos[0], self.box_pos[1] - 1) and \
           self.b_pos == (self.box_pos[0], self.box_pos[1] + 1) and \
           action_list[0] == action_list[1]:
            action_map = self.action_map[action_list[0]]
            self.a_pos = self.move(self.a_pos,
                                  (self.a_pos[0] + action_map[0], self.a_pos[1] + action_map[1]))
            self.b_pos = self.move(self.b_pos,
                                  (self.b_pos[0] + action_map[0], self.b_pos[1] + action_map[1]))
            self.box_pos = self.move(self.box_pos,
                                    (self.box_pos[0] + action_map[0], self.box_pos[1] + action_map[1]))
        else:
            action_map = self.action_map[action_list[0]]
            self.a_pos = self.move(self.a_pos,
                                  (self.a_pos[0] + action_map[0], self.a_pos[1] + action_map[1]))
            action_map = self.action_map[action_list[1]]
            self.b_pos = self.move(self.b_pos,
                                  (self.b_pos[0] + action_map[0], self.b_pos[1] + action_map[1]))

        _a_box = abs(self.a_pos[0] - self.box_pos[0]) + abs(self.a_pos[1] - (self.box_pos[1] - 1))
        _b_box = abs(self.b_pos[0] - self.box_pos[0]) + abs(self.b_pos[1] - (self.box_pos[1] + 1))
        _box_goal = abs(self.box_pos[0] - self.goal_pos[0]) + abs(self.box_pos[1] - self.goal_pos[1])

        if self.box_pos == self.goal_pos:
            reward = 100
            done = True
        else:
            reward = a_box - _a_box + b_box - _b_box + box_goal - _box_goal
            done = False
        a_pos=self.a_pos[0]*7+self.a_pos[1]
        b_pos=self.b_pos[0]*7+self.b_pos[1]
        box_pos=self.box_pos[0]*7+self.box_pos[1]
        return [a_pos, b_pos, box_pos], reward, done, {}

    def render(self):
        obs = np.ones((7 * 20, 7 * 20, 3))
        for i in range(7):
            for j in range(7):
                if self.field[i][j] == 1:
                    cv2.rectangle(obs, (j*20, i*20), (j*20+20, i*20+20), (0, 0, 0), -1)
        cv2.rectangle(obs, (self.a_pos[1]*20, self.a_pos[0]*20), (self.a_pos[1]*20+20, self.a_pos[0]*20+20), (0,0,255), -1)
        cv2.rectangle(obs, (self.b_pos[1]*20, self.b_pos[0]*20), (self.b_pos[1]*20+20, self.b_pos[0]*20+20), (255,0,0), -1)
        cv2.rectangle(obs, (self.box_pos[1]*20, self.box_pos[0]*20), (self.box_pos[1]*20+20, self.box_pos[0]*20+20), (0,255,0), -1)
        cv2.imshow('Move Box', obs)
        cv2.waitKey(100)

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
        self.gamma =0.09
        self.epsilon =0.2


class QLearningAgent:
    def __init__(self, args):
        self.obs_n = args.obs_n
        self.act_n = args.act_n
        self.lr = args.lr
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.Q = np.zeros((args.obs_n, args.obs_n,args.obs_n,args.act_n,args.act_n))

    def select_action(self, obs, if_train=True):
        if(if_train):
            if np.random.rand() <= self.epsilon:
                action_a= np.random.choice(self.act_n)
                action_b= np.random.choice(self.act_n)
                return [action_a,action_b]
            else:
                action= self.Arg_max(obs)
        else:
            action=self.Arg_max(obs)
        return [action[0],action[1]]

    def Arg_max(self,obs):
        Q_list = self.Q[obs[0], obs[1],obs[2], :,:]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)  # maxQ可能对应多个action
        action = np.random.choice(action_list[0].shape[0])
        a=[action_list[0][action],action_list[1][action]]
        return a

    def update(self, transition):
        obs, action, reward, obs1, done = transition
        predict_Q=self.Q[obs[0], obs[1],obs[2],action[0],action[1]]
        if done:
            target_Q=reward
        else:
            target_Q=reward+self.gamma*np.max(self.Q[obs1[0], obs1[1],obs1[2],:])
        self.Q[obs[0], obs[1],obs[2],action[0],action[1]]+=self.lr*(target_Q-predict_Q)

def q_learning_train(args):
    env = args.env
    agent = args.agent
    episodes = args.episodes
    max_steps = args.max_step
    rewards = []
    mean_100ep_reward = []
    for episode in range(episodes):
        episode_reward = 0
        env.reset()
        obs=[env.a_pos[0]*7+env.a_pos[1],env.b_pos[0]*7+env.b_pos[1],env.box_pos[0]*7+env.box_pos[1]]
        for t in range(max_steps):
            action=agent.select_action(obs)
            obs_next,reward,done,_=env.step(action)
            transition=obs, action, reward, obs_next, done
            agent.update(transition)
            obs=obs_next
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
    env.reset()
    obs=[env.a_pos[0]*7+env.a_pos[1],env.b_pos[0]*7+env.b_pos[1],env.box_pos[0]*7+env.box_pos[1]]
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
        time.sleep(1)
        if done:
            break
        print(i)
        if(i>20):
            break
    return total_reward

if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)

    q_learning_args = Arguments()
    env = EnvMoveBox()
    q_learning_args.env = env
    q_learning_args.obs_n = 49
    q_learning_args.act_n = 4
    q_learning_args.agent = QLearningAgent(q_learning_args)

    q_learning_rewards = q_learning_train(q_learning_args)

    q_learning_test_rewards=q_learning_test(q_learning_args)

    print('q_learning_test_reward = %.1f'%(q_learning_test_rewards))

    plt.plot(range(q_learning_args.episodes), q_learning_rewards, label='Q Learning')
    plt.legend()
    plt.show()