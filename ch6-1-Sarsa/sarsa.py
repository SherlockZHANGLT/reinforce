"""
Class Monte Carlo Explore Start-First Visit
Author: Hejun Wu
Last Modification Date: 2020-05
Free to download for students only
"""

import pandas as pd
import numpy as np
import random

import sys
sys.path.append('../')

import gridworldenv as env
import gridgamestates as at


# maximum iterations, can be adjusted according to the need of larger games
MAX_ITERATIONS = 2000  #2000*0.1 = 200
EPSILON = 0.1
MINIMUM_W = 0.0001
ACTION_PROBABILITY = [0.4, 0.2, 0.2, 0.2]
Q_VALUE_RANGE = 20
ALPHA_CONSTANT = 1.0
GAMMA = 0.9
Alpha=0.1

#Class Monte Carlo Epsilon Soft - First Visit
class Sarsa():
    def __init__(self, n_stats, n_acts, para_gamma):
        self.n_states = n_stats             # number of states
        self.n_actions = n_acts             # number of actions
        self.gg = at.GridGame(n_stats, n_acts)    # this object handles interaction with the environment
        self.action_id_num = list(range(n_acts*2))
        # In table q_list, each row corresponds to a state, but the number of columns is double the number of actions.
        # In columns 0 to (num_acts-1), each element records number of times the (s, a) pair has occurred
        # In columns num_acts to (2*num_acts-1), each element records average Return of the (s, a) pair
        self.q_list = pd.DataFrame(columns=self.action_id_num, dtype=np.float64)
        self.C = [[0 for i in range(n_acts)] for i in range(n_stats)]  # off-policy C[s][a]
        self.Pi = list(range(n_stats))      # policy, a list of best action corresponding to state
        self.PiProb = [[0 for i in range(n_acts+1)] for i in range(n_stats)]
        self.gamma = para_gamma

    def append_state(self, sid):          # append state to Q table
        if sid in self.q_list.index:      # if state exists in table, don't append it again
            return
        self.q_list = self.q_list.append(
            pd.Series(                      # append state as a new row
                [0]*self.n_actions*2,       # use 0 as initial value
                index=self.q_list.columns,  # each column of this new row is consistent with the column in the table
                name=sid,
            )
        )

    def append_state_initQ(self, sid, aq_values):  # append state to Q table
        if sid in self.q_list.index:  # if state exists in table, don't append it again
            return
        self.q_list = self.q_list.append(
            pd.Series(  # append state as a new row
                aq_values,  # use 0 as initial value
                index=self.q_list.columns,  # each column of this new row is consistent with the column in the table
                name=sid,
            )
        )

    def print_Qlist(self):
        print(
            "Q: StateID(0..w*h-1)   0-up times   1-down times   2-left times    3- right times    4-up value   5-down value    6-left value     7-right value")
        print(self.q_list)

    def sarsa_initQ(self):
        rqs = [1] * self.n_actions * 2
        for s in range(self.n_states):
            if env.is_terminal_state(s):
                for a in range(self.n_actions):
                    vid = self.get_a_value_idx(a)
                    rqs[vid] = 0
            else:
                for a in range(self.n_actions):
                    vid = self.get_a_value_idx(a)
                    rqs[vid] = random.randint(0,self.n_actions) / random.randint(self.n_actions, self.n_actions * self.n_states)
            self.append_state_initQ(s,rqs)
        self.print_Qlist()

    def get_policy(self):                   # get policy list
        for i in range(len(self.Pi)):
            self.Pi[i]=self.deterministic_pi(i)
        return self.Pi

    # calculate index of average Return value column corresponding to action
    def get_a_value_idx(self, a):
        return a + self.n_actions
    
    # calculate index of times of occurrence column corresponding to action
    def get_a_num_idx(self, a):
        return a

    def deterministic_pi(self, s):
        max_a = -1
        nn=2
        for a in range(self.n_actions):
            n_idx = self.get_a_num_idx(a)
            if(self.q_list.loc[s, n_idx]):
                v_idx = self.get_a_value_idx(a)
                if(max_a == -1):
                    max_q = self.q_list.loc[s, v_idx]
                    max_a = a
                elif (max_q < self.q_list.loc[s, v_idx]):
                    max_a = a
                    max_q = self.q_list.loc[s, v_idx]
                elif (max_q == self.q_list.loc[s, v_idx]):
                    i = random.randint(1,nn)
                    nn=nn+1
                    if(i==1):
                        max_a = a
                        max_q = self.q_list.loc[s, v_idx]
        if max_a == -1:
            max_a =0
        if max_a >= self.n_actions:
            max_a = 0
        a_list=[]
        for a in range(self.n_actions):
            n_idx = self.get_a_num_idx(a)
            if(self.q_list.loc[s, n_idx]):
                v_idx = self.get_a_value_idx(a)
                if (max_q-self.q_list.loc[s, v_idx]<=0.3 and self.q_list.loc[s, a]>40):
                    a_list.append(a)
        if(len(a_list)==2):
            max_a=a_list[0]+a_list[1]+4
            if(a_list[0]==0 or a_list[1]==0):
                max_a=max_a-1
        elif(len(a_list)==3):
            max_a=a_list[0]+a_list[1]+a_list[2]+7
        elif(len(a_list)==4):
            max_a=14
        return max_a

    def get_pi_bp_ratio(self, s, pi_a, prob):
        if (self.PiProb[s][self.n_actions]):
            pi = self.PiProb[s][pi_a] / self.PiProb[s][self.n_actions]
        else:
            pi = 1
        return pi/float(prob)


    def epsilon_greedy_pi(self, s):
        ra = random.randint(0, self.n_actions-1) # random action according to epsilon
        a = self.get_argmax_action(s)
        ep = random.randint(0, 100)
        if(ep > EPSILON * 100):
            return a
        return ra # falling into epsilon

    # add a new Return value of (s, a) pair and recalculate average Return
    def calc_avg_return(self, s, a, r):
        if(a == -1):    # a == -1 means s is a terminal state
            for i in range(self.n_actions):     # record this Return value for all actions on terminal state
                n_idx = self.get_a_num_idx(i)
                v_idx = self.get_a_value_idx(i)
                self.q_list.loc[s, n_idx] = 1
                self.q_list.loc[s, v_idx] = r
            return
        n_idx = self.get_a_num_idx(a)
        v_idx = self.get_a_value_idx(a)
        n = self.q_list.loc[s, n_idx]
        avg = self.q_list.loc[s, v_idx]
        if n <= 0:      # (s, a) never occurred before
            n = 0
            avg = r
        else:           # (s, a) occurred before, recalculate average
            avg = float(avg * n + r) / (n + 1)
        self.q_list.loc[s, v_idx] = avg
        self.q_list.loc[s, n_idx] = n + 1       # increase times of occurrence

    # get the action which leads to the highest Return on state s
    def get_argmax_action(self, s):
        max_a = -1
        nn=2
        for a in range(self.n_actions):
            n_idx = self.get_a_num_idx(a)
            if(self.q_list.loc[s, n_idx]):
                v_idx = self.get_a_value_idx(a)
                if(max_a == -1):
                    max_q = self.q_list.loc[s, v_idx]
                    max_a = a
                elif (max_q < self.q_list.loc[s, v_idx]):
                    max_a = a
                    max_q = self.q_list.loc[s, v_idx]
                elif (max_q == self.q_list.loc[s, v_idx]):
                    i = random.randint(1,nn)
                    nn=nn+1
                    if(i==1):
                        max_a = a
                        max_q = self.q_list.loc[s, v_idx]
        if max_a == -1:
            max_a =0
        if max_a >= self.n_actions:
            max_a = 0
        return max_a

    def get_Q_value(self, sid, aid):
        v_idx = self.get_a_value_idx(aid)
        n_idx = self.get_a_num_idx(aid)
        Q = self.q_list.loc[sid, v_idx]
        Q_n = self.q_list.loc[sid, n_idx]
        return Q, Q_n

    def set_Q_value(self, sid, aid, Q):
        v_idx = self.get_a_value_idx(aid)
        n_idx = self.get_a_num_idx(aid)
        self.q_list.loc[sid, v_idx] = Q
        self.q_list.loc[sid, n_idx] += 1
        return self.q_list.loc[sid, n_idx]

    # 用于从待选动作中均匀的选取某个动作，其中alist[]列表中的动作要排除掉（在正态分布中，把居中的和两边的动作排除掉，因为它们已经被选过）
    # 在仅有四个动作的智能体上，本函数实际上只有一个选择，本函数的安排主要是为了以后扩展到更多动作的智能体上。


    def normal_distribution_action(self, state):
        mg, ma = state.get_gain_value()
        sa = (ma + self.n_actions - 1) % self.n_actions
        ga = (ma + self.n_actions + 1) % self.n_actions
        prob = 0
        r = random.randint(0,100)
        if r >= 100 * (1- ACTION_PROBABILITY[0]): #mean 0.4
            return ACTION_PROBABILITY[prob], ma
        if r >= 20:
            if(r<40):
                prob += 1
                return ACTION_PROBABILITY[prob], sa
            prob += 2
            return ACTION_PROBABILITY[prob], ga
        prob += 3
        pair = (sa, ma, ga)
        ra = state.get_random_remain_actions(pair)
        return ACTION_PROBABILITY[prob],ra
# ALGORITHM: off-policy Monte Carlo E
# -First Visit is hiden in episode generation procedure in game-state class.
    def sarsa_learn(self, n_actions, id_bpolicy):
        steps = 0
        self.sarsa_initQ()
        while True:
            S = self.gg.get_random_state_id() #get a state randomly, ID is S
            A = self.epsilon_greedy_pi(S)  # the epsilon greedy policy to choose an action, as Q has values aAlphaeady
            self.Pi[S] = A
            while not env.is_terminal_state(S):
                S_P = self.gg.get_next_state_id(S,A)
                R = env.get_reward_observation(S, S_P)
                A_P = self.epsilon_greedy_pi(S_P)
                Q_sa, n_q = self.get_Q_value(S, A)
                Q_spap, n_pq = self.get_Q_value(S_P, A_P)
                Q_sa = Q_sa + Alpha*(R + GAMMA * Q_spap - Q_sa)
                self.set_Q_value(S, A, Q_sa)
                S = S_P
                A = A_P
                self.Pi[S] = A
            steps += 1
            if steps > MAX_ITERATIONS:
                break
        # Show the resulted policiy and Q table
        print("Policy: " + str(self.Pi))
        self.print_Qlist()









