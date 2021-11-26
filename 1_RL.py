import gym
import numpy as np
from matplotlib import pyplot as plt

gamma=0.8

class CliffWalking:
    def __init__(self):
        self.actions = (0, 1, 2, 3)
        self.rewards = [[-1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1, -1],
                        [-1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1, -1],
                        [-1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1, -1],
                        [-1, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,  0]]

    def step(self, pos, a):
        i, j = pos
        if a == 0:#向上
            i_ = i-1 if i > 0 else 0
            j_ = j
        elif a == 1:#向右
            i_ = i
            j_ = j+1 if j < 11 else j
        elif a == 2:#向下
            i_ = i+1 if i < 3 else i
            j_ = j
        elif a == 3:#向左
            i_ = i
            j_ = j-1 if j > 0 else j
        return i_, j_, self.rewards[i_][j_]


class PolicyIteration:
    def __init__(self, env):
        self.env = env
        self.PI = np.array([[np.random.choice((0,1,2,3)) for j in range(12)] for j in range(4)])
        self.V = np.array([[np.random.random() for j in range(12)] for j in range(4)])
        self.V[-1][-1] = 0

    def learn(self):
        for i in range(100):
            self.policy_evaluation()
            self.policy_imporvement()
        print(self.PI)

    def policy_evaluation(self):
        for i in range(1000):
            delta=0.0
            for i in range(4):
                for j in range(12):
                    action=self.PI[i][j]
                    i_,j_,r=self.env.step((i,j),action)
                    new_v=r+gamma*self.V[i_][j_]
                    delta=abs(self.V[i][j]-new_v) if abs(self.V[i][j]-new_v)>delta else delta
                    self.V[i][j]=new_v
            if delta<1e-6:
                break

    def policy_imporvement(self):
        for i in range(4):
            for j in range(12):
                a1=self.env.actions[0]
                i_,j_,r=self.env.step((i,j),a1)
                v1=r+gamma*self.V[i][j]
                for action in self.env.actions:
                    i_,j_,r=self.env.step((i,j),action)
                    if v1<r+gamma*self.V[i_][j_]:
                        a1=action
                        v1=r+gamma*self.V[i_][j_]
                self.PI[i][j]=a1



class ValueIteration:
    def __init__(self, env):
        self.env = env
        self.PI = np.array([[np.random.choice((0,1,2,3)) for j in range(12)] for j in range(4)])
        self.V = np.array([[np.random.random() for j in range(12)] for j in range(4)])
        self.V[-1][-1] = 0

    def learn(self):
        for t in range (1000):
            delta=0.0
            for i in range(4):
                for j in range(12):
                    a1=self.env.actions[0]
                    i_,j_,r=self.env.step((i,j),a1)
                    v1=r+gamma*self.V[i][j]
                    for action in self.env.actions:
                        i_,j_,r=self.env.step((i,j),action)
                        if v1<r+gamma*self.V[i_][j_]:
                            a1=action
                            v1=r+gamma*self.V[i_][j_]
                    delta=abs(v1-self.V[i][j]) if abs(v1-self.V[i][j])>delta else delta
                    self.PI[i][j]=a1
                    self.V[i][j]=v1
            if delta<1e-6:
                break
        print(self.PI)


if __name__ == '__main__':
    np.random.seed(0)
    env = CliffWalking()

    PI = PolicyIteration(env)
    PI.learn()

    VI = ValueIteration(env)
    VI.learn()