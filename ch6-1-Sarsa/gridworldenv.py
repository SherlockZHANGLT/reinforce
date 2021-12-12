"""
Class GRID_WORLD environment for RL
Author: Hejun Wu
Reference: GRID_WORLD on Internet
Last Modification Date: 2020-05
Free to download for students only
"""

import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
from enum import Enum
from gridgamestates import GridStates


#-----------------------------------------------------------------------------------------------------------
UNIT = 80   # (GUI) pixels of grid side length
GRID_WORLD_H = 4  # GRID_WORLD height (number of grid)
GRID_WORLD_W = 4  # GRID_WORLD width (number of grid)
GRID_WORLD_DIST = 40      # (GUI) if draw two GRID_WORLDs, specify the distance between the first GRID_WORLD and the second GRID_WORLD
START_X = 10        # (GUI) x axis offset (left margin)
START_Y = 10        # (GUI) y axis offset (top margin)
SECOND_M_X = START_X + GRID_WORLD_H * UNIT + GRID_WORLD_DIST    # x axis offset of the second GRID_WORLD
SECOND_M_Y = START_Y                                # y axis offset of the second GRID_WORLD
GRID_HALF = UNIT / 2    # (GUI) half pixel of grid side length
ACTION_IDS = ["up", "down",  "left",  "right"]#,"ud","ul","ur","dl","dr","lr","udl","udr","ulr","dlr","all"
PENALTY = -2
symbol=['|','⭠⭡', '⭡⭢','⭠⭣','⭣⭢','⭠⭢','⭠|', '|⭢','⭠⭡⭢','⭠⭣⭢', '+',]
#--------------------------------------------------------------------------------------------------------------

# enironment and GUI
class GridWorld(tk.Tk, object):
    def __init__(self):
        super(GridWorld, self).__init__()
        self.n_charts = 2   # (GUI) number of charts to demonstrate the values and the actions on the states of the grid world
        self.s_action_ids = ACTION_IDS  # list of actions
        self.n_actions = len(self.s_action_ids)  # number of actions
        self.title('GRID_WORLD')  # GUI window title
        self.geometry('{0}x{1}'.format(START_X * 2 + GRID_WORLD_W * UNIT * self.n_charts  + GRID_WORLD_DIST,
                                       START_Y * 2 + GRID_WORLD_H * UNIT))  # GUI window size
        #self._build_GRID_WORLD()  # draw GUI GRID_WORLD
        self.t_origin = np.array([START_X + GRID_HALF, START_X + GRID_HALF])  # The origin of the first chart
        self.s_t_origin = np.array([SECOND_M_X + GRID_HALF, SECOND_M_Y + GRID_HALF])  # The oringin of the second chart


    # 画窗口和网格
    def draw_window(self, n_charts):  # draw GUI GRID_WORLD
        self.n_charts = n_charts
        self.canvas = tk.Canvas(self, bg='white',
                                height=START_Y + GRID_WORLD_H * UNIT * self.n_charts  + GRID_WORLD_DIST,
                                width=START_X + GRID_WORLD_W * UNIT * self.n_charts  + GRID_WORLD_DIST * self.n_charts )

        # create grids------------------------------------------S-----------------------------
        c = GRID_WORLD_W * UNIT
        r = 0

        for c in range(START_X, START_X + GRID_WORLD_W * UNIT + UNIT, UNIT):  # (GUI) draw vertical lines
            x0, y0, x1, y1 = c, START_Y, c, START_Y + GRID_WORLD_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)  # (GUI) line is from (xo, y0) to (x1, y1)

        for r in range(START_Y, START_Y + GRID_WORLD_H * UNIT + UNIT, UNIT):  # (GUI) raw horizontal lines
            x0, y0, x1, y1 = START_X, r, START_X + GRID_WORLD_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)  # (GUI) line is from (xo, y0) to (x1, y1)

        # self.n_charts  = 2　is used in dynamic programming, two grid worlds to indicate the previous value and the current value
        if (self.n_charts == 2):  # (GUI) draw the second GRID_WORLD
            for d in range(SECOND_M_X, SECOND_M_X + GRID_WORLD_W * UNIT + UNIT, UNIT):  # (GUI) draw vertical lines
                x0, y0, x1, y1 = d, SECOND_M_Y, d, SECOND_M_Y + GRID_WORLD_H * UNIT
                self.canvas.create_line(x0, y0, x1, y1)  # (GUI) line is from (xo, y0) to (x1, y1)
            for q in range(SECOND_M_Y, SECOND_M_Y + GRID_WORLD_H * UNIT + UNIT, UNIT):  # (GUI) raw horizontal lines
                x0, y0, x1, y1 = SECOND_M_X, q, SECOND_M_X + GRID_WORLD_W * UNIT, q
                self.canvas.create_line(x0, y0, x1, y1)  # (GUI) line is from (xo, y0) to (x1, y1)
            self.rect = self.canvas.create_rectangle(  # (GUI) draw a rectangle as border of the two GRID_WORLDs
                START_X, START_Y,
                SECOND_M_X + GRID_WORLD_W * UNIT, SECOND_M_Y + GRID_WORLD_H * UNIT
            )
        # create origin
        else:
            self.rect = self.canvas.create_rectangle(  # (GUI) draw a rectangle as border of the GRID_WORLD
                START_X, START_Y,
                START_X + GRID_WORLD_W * UNIT, START_Y + GRID_WORLD_H * UNIT
            )
        self.canvas.pack()

    # timer, need in dynamic programming
    def reset(self):
        time.sleep(0.5)

    # 网格世界宽度
    def get_width(self):
        return GRID_WORLD_W

    # 网格世界高度
    def get_height(self):
        return GRID_WORLD_H

    # 动作数
    def get_n_actions(self):
        return len(self.s_action_ids)

    # 状态数
    def get_n_states(self):
        return GRID_WORLD_H * GRID_WORLD_W

    # 边界
    def get_border(self):
        return (GRID_WORLD_W, GRID_WORLD_H)

    # 各种动作的图标
    def get_action_arrow(self, s, action):  # (GUI) get symbols corresponding to actions
        if is_terminal_state(s):
            return '⭯'
        if action > 3:
            return symbol[action-4]
        options = {
            self.s_action_ids.index('up'): '⭡',
            self.s_action_ids.index('down'): '⭣',
            self.s_action_ids.index('left'): '⭠',
            self.s_action_ids.index('right'): '⭢'
        }
        return options[action]

    # 展示结果策略 draw the policy on the grid
    def fill_pi(self, pi):  # (GUI) draw action by policy in corresponding grid
        for s in range(self.get_n_states()):
            a = pi[s]
            t = self.get_action_arrow(s, a)  # (GUI) get symbols corresponding to actions
            tx, ty = get_state_pos(s)
            cx = tx * UNIT + self.t_origin[0]
            cy = ty * UNIT + self.t_origin[1]
            self.canvas.create_text(cx, cy, font=("Helvetica", 14), text=t, fill="red")  # (GUI) draw the symbol

    def filltext(self, tx, ty, t, prime=0):
        if prime:
            cx = tx * UNIT + self.s_t_origin[0]
            cy = ty * UNIT + self.s_t_origin[1]
        else:
            cx = tx * UNIT + self.t_origin[0]
            cy = ty * UNIT + self.t_origin[1]

        rect = self.canvas.create_rectangle(
            cx - GRID_HALF, cy - GRID_HALF,
            cx + GRID_HALF, cy + GRID_HALF, fill='white')
        if (tx == 0 and ty == 0) or (tx == GRID_WORLD_W - 1 and ty == GRID_WORLD_H - 1):
            rect = self.canvas.create_rectangle(
                cx - GRID_HALF, cy - GRID_HALF,
                cx + GRID_HALF, cy + GRID_HALF,
                outline='red')
            # fill='white'
        else:
            rect = self.canvas.create_rectangle(
                cx - GRID_HALF, cy - GRID_HALF,
                cx + GRID_HALF, cy + GRID_HALF,
                outline='blue')  # fill='white'
        self.canvas.create_text(cx, cy, font=("Helvetica", 14), text=round(t, 1))

    def fillvalues(self, values1, values2):
        for x in range(GRID_WORLD_W):
            for y in range(GRID_WORLD_H):
                self.filltext(x, y, values1[x][y])
                self.filltext(x, y, values2[x][y], 1)
        time.sleep(0.1)

    def draw_action(self, tx, ty, t):
        cx = tx * UNIT + self.t_origin[0]
        cy = ty * UNIT + self.t_origin[1] - GRID_HALF / 2
        self.canvas.create_text(cx, cy, font=("Helvetica", 14), text= t)

    def fill_pi_list(self, pi_list):
        for x in range(GRID_WORLD_W):
            for y in range(GRID_WORLD_H):
                sid = get_state_id_via_pos((x, y))
                t = self.get_action_arrow(sid, pi_list[x][y])
                self.draw_action(x, y, t)
        time.sleep(0.001)

    def render(self):
        time.sleep(0.3)
        self.update()
#--------------------------------------------------------------------------------------------------------
#The next part is global functions of the environment
#reward is 1 in terminal state and -1 in other states
"""-------------------------------------------------------------------------------------------------------
                     [ 1,-1,-1,-1],
                     [-1,-1,-1,-1],
                     [-1,-1,-1,-1],
                     [-1,-1,-1, 1],
"""

#设置窗口上图表数量，DP用2， MC用1： Dynamic programming needs two Monte Carlo nees one.
def set_num_charts(n_charts):
    g_n_charts = n_charts

#动作种类的字符串
def get_action_str():
    return ACTION_IDS
#全局观察奖励
#global func to observe the rewards

def get_reward_observation(csid, nid):
    if (is_terminal_state(csid)):
        reward = 1
    elif csid == nid:
        reward = PENALTY
    else:
        reward = -1
    return reward

#get number of actions
def get_n_actions():
    return len(ACTION_IDS)

# 判断是否是终止态
def is_terminal_state(s):
    if s == 0 or s == GRID_WORLD_H * GRID_WORLD_W -1:   # the first or the last grid are terminal states
        return True
    return False

#GridWorld 边界
def get_border():
    return GRID_WORLD_W, GRID_WORLD_H

#从坐标转换状态ID
def get_state_id_via_pos(pos):  # convert coordinates to a corresponding id value
    x, y = pos[0], pos[1]
    sid = y * GRID_WORLD_W + x
    return sid

#从状态ID获取坐标
def get_state_pos(sid):         # convert a id value to corresponding coordinates
    x = sid % GRID_WORLD_W
    y = (int) (sid / GRID_WORLD_W)
    return (x, y)

# next state after moving up
def get_up_state(sid, act):     # return next state taking action 'up'
    coord = get_state_pos(sid)
    x = coord[0]
    y = coord[1]
    if (y > 0):
        y -= 1
    pos = (x, y)
    nid = get_state_id_via_pos(pos)
    return nid

# next state after moving down
def get_down_state(sid, act):   # return next state taking action 'down'
    coord = get_state_pos(sid)
    x = coord[0]
    y = coord[1]
    if (y < GRID_WORLD_H -1 ):
        y += 1
    pos = (x, y)
    nid = get_state_id_via_pos(pos)
    return nid

# next state after moving left
def get_left_state(sid, act):   # return next state taking action 'left'
    coord = get_state_pos(sid)
    x = coord[0]
    y = coord[1]
    if (x > 0 ):
        x -= 1
    pos = (x, y)
    nid = get_state_id_via_pos(pos)
    return nid

# next state after moving right
def get_right_state(sid, act):  # return next state taking action 'right'
    coord = get_state_pos(sid)
    x = coord[0]
    y = coord[1]
    if (x < GRID_WORLD_W -1 ):
        x += 1
    pos = (x, y)
    nid = get_state_id_via_pos(pos)
    return nid

# get_state_after_action 函数中的switcher碰到错误动作，会进入本函数
# default option of the switcher in get_state_after_action
def invalid(state, act):
    print(str(state)+': [{}] is an invalid action'.format(act))
    return 0

#return the id of next state that is the result of action
def peek_state_after_action(sid, act):
    n_actions = get_n_actions()
    if(act >= n_actions):
        return sid
    switcher = {
        ACTION_IDS.index('up'): get_up_state,
        ACTION_IDS.index('down'): get_down_state,
        ACTION_IDS.index('left'): get_left_state,
        ACTION_IDS.index('right'): get_right_state,
    }
    func = switcher.get(act, lambda state, act: invalid(sid, act))
    return func(sid, act)
#--------------------------------------------------------------------------------------------------------
