"""
Class Grid games and states for RL
Author: Hejun Wu
Reference: Maze on Internet
Last Modification Date: 2020-05
Free to download for students only
"""
import numpy as np
import pandas as pd
import gridworldenv as env
import random
from datetime import datetime


MEDIAN_ACTION_NUM = 3


# GridGame handles the interaction with the environment for algorithm
# can be extended to other algorithms
class GridGame():
    #格子游戏所需的数据 Maintain data in grid game
    # use number of states, number of actions, and action types to init a GridGameclass object
    def __init__(self, n_stats, n_acts):
      #  self.action_num = list(range(n_acts))
        random.seed(datetime.now())
        self.border = env.get_border()          # a tuple (WIDTH, HEIGHT) of environment to avoid an agent to go out of the border
        self.n_actions = n_acts                 # number of actions
        self.all_states=[]                      # list of all states of the game
        for i in range(n_stats):
            x = i % self.border[0]
            y = (int)(i / self.border[0])
            self.all_states.append(GridStates(x, y, self.n_actions))

    # explore a random start
    def get_random_start(self):
        is_terminal = True
        sid = 0
        while is_terminal:                      # regenerate if a terminal state is generated
            x = random.randint(0,self.border[0]-1)
            y = random.randint(0, self.border[1]-1)
            pos = (x, y)
            sid = env.get_state_id_via_pos(pos)
            if env.is_terminal_state(sid):
                is_terminal = True
            else:
                is_terminal = False
        return self.all_states[sid]


    def get_random_state_id(self):
        is_terminal = True
        sid = 0
        while is_terminal:                      # regenerate if a terminal state is generated
            x = random.randint(0,self.border[0]-1)
            y = random.randint(0, self.border[1]-1)
            pos = (x, y)
            sid = env.get_state_id_via_pos(pos)
            if env.is_terminal_state(sid):
                is_terminal = True
            else:
                is_terminal = False
        return sid

    # next state after moving right
    def get_random_action(self, s, n_actions):
        if env.is_terminal_state(s):
            return -1
        r_num = random.randint(0, n_actions-1)
        return r_num

    # func to calculate the gains
    def calc_returns(self, csid, action):
        cur_state = self.all_states[csid]
        next_state = self.get_next_state(cur_state, action)
        if (env.is_terminal_state(csid)):
            gain = 1
            for a in range(self.n_actions):
                cur_state.set_gain_value(a, gain)
        else:
            g, a = next_state.get_gain_value()
            gain = -1 + g
            # record gain of this state, so that previous state of this state can get gain of this state
            cur_state.set_gain_value(action, gain)
        return gain

    #看下一个状态，不改变状态表的数据，只返回状态的ID。return the id of the next state that is the result of action
    def get_next_state(self, state, action):
        sid = env.get_state_id_via_pos(state.get_state_data())
        ps =  self.all_states[sid]
        if env.is_terminal_state(sid):      # if state is terminal, return it as is
            return ps
        else:                               # if not, take the action and return the next state
            nid = env.peek_state_after_action(sid, action)
            return self.all_states[nid]

        # 看下一个状态，不改变状态表的数据，只返回状态的ID。return the id of the next state that is the result of action
    def get_next_state_id(self, sid, aid):
        if env.is_terminal_state(sid):  # if state is terminal, return it as is
            return sid
        else:  # if not, take the action and return the next state
            nid = env.peek_state_after_action(sid, aid)
            return nid

    #生成一个给定起始状态的动作完整集 episode， 这个起始点是算法给定的。
    # generate an episode with start state and action， given by the Monte-Carlo algorithm
    def generate_episode_policy(self, start, action, bpolicy_func):
        steps = 0
        cur_state = start
        a = action
        episode = []                # 用于存放episode。 Records trajectory of this episode
        s = env.get_state_id_via_pos(cur_state.get_state_data())
        while(not env.is_terminal_state(s)):
            # 一直把状态-动作对加入列表，直到终止态。 Keep appending new state-actions, until reach the terminal state.
            # 要用状态ID， 把格子的坐标转换成ID。 Use the state ID, so we need to convert the coordinates to ID.
            # 状态-动作对中，动作是随机的，这样下一个状态也是随机的  Action is random, so that the next state is random
            s = env.get_state_id_via_pos(cur_state.get_state_data())
            if(bpolicy_func == 0):
                a = self.get_random_action(s, self.n_actions)
                prob = 0.25
            else:
                prob, a = bpolicy_func(cur_state)
            #课件算法Monte Carlo ES (Exploring Starts)第9行，为第11行做准备。  Line-9 of onte Carlo ES (Exploring Starts)
            # ALGORITHM---L9-10 G: returns follow occurences of s, a.
            #课件算法Monte Carlo ES (Exploring Starts)第10行 ：
            # ALGORITHM---Line-10 G: returns follow the first occurance of s, a. and APPEND G to
            G = -1 #临时的， 因为本函数不参与算法的计算。 real gain will be added in Monte Carlo algorithm (mc-*.py)
            next_state = self.get_next_state(cur_state, a)
            episode += [(s, a, G, prob)]
            cur_state = next_state
            steps += 1
            if steps >= 100000:
                raise Exception("Exception raised, because program got stuck in MC Qepisode generation...\n")
        return episode

        #以下三元素，s, a, G 属于终止态
        #终止态无动作
        # 终止态的收益是1
        #把终止态加入到episode,生成一个episode  Monte Carlo algorithm needs complete episodes for sampling.

class GridStates():
    #状态类的初始化，x,y是本状态坐标， n_acts是动作种类， gain初始化为负极值
    def __init__(self, x, y, n_acts):
        self.coordinates = [x,y]
        self.n_actions = n_acts
        self.gain = [-0xfffff]*n_acts
        self.actions = [-1] * n_acts
        self.chosen_actions = [0] * (n_acts - MEDIAN_ACTION_NUM)

    #获取本状态的数据，Grid状态即为其坐标值
    def get_state_data(self):
        return self.coordinates[0], self.coordinates[1]

    # 设置本状态的数据，一般不用，因初始化时状态数据已经设置好
    def set_state_data(self, coord):
        self.coordinates[0] = coord[0]
        self.coordinates[1] = coord[1]

    #本函数给出本状态下记录的最高的奖励
    def get_gain_value(self):
        gain = self.gain[0]
        max_a = 0
        for a in range(1,self.n_actions):
            if gain < self.gain[a]:
                gain = self.gain[a]
                max_a = a
        return gain, max_a

    #用表格记录某个动作的奖励
    def set_gain_value(self, action, gain):
        self.gain[action] = gain

    def get_random_remian_actions(self, alist):
        chosen = 0
        for a in range(len(alist)):
            self.actions[alist[a]] = 1
        for a in range(self.n_actions):
            if self.actions[a] == -1:
                self.chosen_actions[chosen] = a
                chosen += 1
        r = random.randint(0, chosen - 1)
        choice = self.chosen_actions[r]
        for a in range(len(alist)):
            self.actions[alist[a]] = -1
        return choice


