"""
Main program for running
Author: Hejun Wu
Reference: Maze on Internet
Last Modification Date: 2020-05
Free to download for students only
"""

from sarsa import Sarsa
import pandas as pd
import numpy as np

import sys
sys.path.append('../')
import gridworldenv as env
from gridworldenv import GridWorld


VALUE_LIST_0 = 0
VALUE_LIST_PRIME = 1

if __name__ == "__main__":

    # Maze is the environment
    gw = GridWorld()                         # initialize environment
    gw.draw_window(1)

    #从环境中获取状态、动作
    #n_states: number of states
    #s_action_ids: action str such as moving up, down, ...\
    #n_actions: number of actions
    n_states = gw.get_n_states()
    n_actions = gw.get_n_actions()

    #Monte Carlo Explore Start First - Visit / or every - visit(optional)
    m_sarsa = Sarsa(n_states, n_actions, 0.9)    # initialize MCES agent, gamma = 0.9
    m_sarsa.sarsa_learn(n_actions, 1)        # learn from environment using off-policy MC, generate a policy

    #从算法中获取策略，再把策略用图形展示出来
    pi = m_sarsa.get_policy()
    gw.fill_pi(pi)                      # draw the policy pi in GUI

    #window, to enable users see the picture
    gw.mainloop()                       # GUI window loop


