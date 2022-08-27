from copy import deepcopy
from distutils.dist import Distribution
from tdw.quaternion_utils import QuaternionUtils
from icecream import ic
import numpy as np
import math


MAGNEBOT_RADIUS: float = 0.22
OCCUPANCY_CELL_SIZE: float = (MAGNEBOT_RADIUS * 2) + 0.05

def l2_dis(x1, x2, y1, y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5

def eular_yaw(qua):
    angle = QuaternionUtils.quaternion_to_euler_angles(qua)[1]
    if abs(qua[3]) < 0.7071068:
        if qua[1] > 0:
            angle = 180-angle
        elif qua[1] < 0:
            angle = -180-angle
    while abs(angle) > 180:
        if angle < 0: angle += 360
        else: angle -= 360
    return angle

def any_ongoing(bots):
    for bot in bots:
        if bot._check_ongoing():
            return True
    return False

def a_star_search(grid: list, begin_point: list, target_point: list, start_pos):
    pos = [start_pos[0], start_pos[2]]
    actions = []
    def cost(inp):
        return inp[0]
    # the cost map which pushes the path closer to the goal
    grid = deepcopy(grid)
    grid[grid==-1] = 1

    #grid[begin_point[0], begin_point[1]] = 0
    
    grid[target_point[0], target_point[1]] = 0
    if grid[begin_point[0], begin_point[1]] == 1:
        begin_point[0] += 1
        if grid[begin_point[0], begin_point[1]] == 1:
            begin_point[1] += 1

    heuristic = np.zeros_like(grid)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            heuristic[i][j] = abs(i - target_point[0]) + abs(j - target_point[1])
            if grid[i][j] == 1:
                heuristic[i][j] = 99

    delta = [[-1, 0],  # go up
             [0, -1],  # go left
             [1, 0],  # go down
             [0, 1]]  # go right

    close_matrix = np.zeros_like(grid)
    close_matrix[begin_point[0]][begin_point[1]] = 1
    action_matrix = np.ones_like(grid) * (8)

    x = begin_point[0]
    y = begin_point[1]
    g = 0
    f = g + heuristic[begin_point[0]][begin_point[1]]
    cell = [[f, g, x, y]]

    found = False  # flag that is set when search is complete
    resign = False  # flag set if we can't find expand

    while not found and not resign:
        if len(cell) == 0:
            resign = True
            return None, None
        else:
            cell.sort(key=cost, reverse=True)
            next = cell.pop()
            x = next[2]
            y = next[3]
            g = next[1]
            f = next[0]

            if x == target_point[0] and y == target_point[1]:
                found = True
            else:
                # delta have four steps
                for i in range(len(delta)):  # to try out different valid actions
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    if x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0]):  # 判断可否通过那个点
                        if close_matrix[x2][y2] == 0 and grid[x2][y2] == 0:
                            g2 = g + 1
                            f2 = g2 + heuristic[x2][y2]
                            cell.append([f2, g2, x2, y2])
                            close_matrix[x2][y2] = 1
                            action_matrix[x2][y2] = i
 
    invpath = []
    x = target_point[0]
    y = target_point[1]
    distrilled_actions = []
    counts = []
    action_counts = 0
    previous_action = None
    invpath.append([x, y])  # we get the reverse path from here
    while x != begin_point[0] or y != begin_point[1]:
        x2 = x - delta[action_matrix[x][y]][0]
        y2 = y - delta[action_matrix[x][y]][1]
        actions.append(action_matrix[x][y])
        x = x2
        y = y2
        invpath.append([x, y])
        

    for action in reversed(actions):
        if previous_action is not None:
            if action == previous_action:
                action_counts += 1
                if action_counts == 3:
                    distrilled_actions.append(action)
                    counts.append(3)
                    action_counts = 0
            else:
                if not(action_counts == 0):
                    counts.append(action_counts)
                    action_counts = 0
                    distrilled_actions.append(previous_action)
                previous_action = action
                action_counts += 1
        else:
            previous_action = action
            action_counts += 1
    
    if not(action_counts == 0):
        distrilled_actions.append(previous_action)
        counts.append(action_counts)
    np.set_printoptions(threshold=np.inf)
    
    if len(distrilled_actions) == 0:
        return None, None
    return distrilled_actions[0], counts[0]


def pos_to_grid(x,z, bound):
    #ic(x,z)
    i = (x - bound.x_min) / (OCCUPANCY_CELL_SIZE)
    j = (z - bound.z_min) / (OCCUPANCY_CELL_SIZE)
    #ic(i,j)
    return [int(round(i)), int(round(j))]
    
def grid_to_pos(i: int, j: int, bound):
    x = bound.x_min + (i * OCCUPANCY_CELL_SIZE)
    z = bound.z_min + (j * OCCUPANCY_CELL_SIZE)
    return np.array([x, 0, z])