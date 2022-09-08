from copy import deepcopy
from distutils.dist import Distribution
from tdw.quaternion_utils import QuaternionUtils
from icecream import ic
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import random
import os
from matplotlib import pyplot as plt

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

def a_star_search(grid: list, begin_point: list, target_point: list, start_pos, return_path = False):
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
                    distrilled_actions.append(previous_action)
                    action_counts = 0
                previous_action = action
                action_counts += 1
        else:
            previous_action = action
            action_counts += 1
    
    if not(action_counts == 0):
        distrilled_actions.append(previous_action)
        counts.append(action_counts)

    #np.set_printoptions(threshold=np.inf)

    if len(distrilled_actions) == 0:
        return None, None

    if return_path:
        return [ele for ele in reversed(invpath)]
    return distrilled_actions[0], counts[0]
    '''new_dis = []
    new_counts = []

    previous_action = distrilled_actions[0]
    for idx in range(1,len(distrilled_actions)):
        if distrilled_actions[idx] == previous_action and counts[idx] == 1:
            pass
        else:
            new_dis.append(distrilled_actions[idx])
            new_counts.append(counts[idx])
        previous_action = distrilled_actions[idx]
    
    if len(new_dis) == 0:
        return None, None'''
    return new_dis[0], new_counts[0]


def pos_to_grid(x,z, bound):
    #ic(x,z)
    i = (x - bound.x_min) / (OCCUPANCY_CELL_SIZE)
    j = (z - bound.z_min) / (OCCUPANCY_CELL_SIZE)
    #ic(i,j)
    return [int(round(i)), int(round(j))]
    
def grid_to_pos(i: int, j: int, bound):
    x = bound.x_min + ((i) * OCCUPANCY_CELL_SIZE)
    z = bound.z_min + ((j) * OCCUPANCY_CELL_SIZE)
    return np.array([x, 0, z])

def eular_to_quat(eular):
    ma = R.from_euler('xyz', eular, degrees=True)
    return ma.as_quat()


class Node():
    def __init__(self, pos):
        self.x = pos[0]
        self.y = pos[1]
        self.parent = None
 
 
class rrtPlanner():
 
    def __init__(self, map, bound, step_length=0.6, rng=None):
        self.step_length = step_length
        self.bound = bound
        self.sample_rate = 0.05
        self.max_iter = 10000
        self.map = map
        self.nodes = list()
        self.rng = rng
 
    def set_points(self, start, goal):
        self.start = Node(start)
        self.goal = Node(goal)
        self.nodes.append(self.start)

    def get_random_node(self):
        node_x = self.rng.uniform(self.bound.x_min, self.bound.x_max)
        node_y = self.rng.uniform(self.bound.z_min, self.bound.z_max)
        node = [node_x, node_y]
 
        return node
 
    def get_nearest_node(self, node_list, rnd):
        d_list = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 for node in node_list]
        min_index = d_list.index(min(d_list))
        return node_list[min_index], min_index
 
    def collision_check(self, new_node):
        pos = [new_node.x, new_node.y]
        grid = pos_to_grid(pos[0], pos[1], self.bound)
        ans = 0
        try: 
            ans += not(self.map[grid[0], grid[1]] == 0)
        except: ans = 1
        dis = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2 for node in self.nodes]
        if min(dis) < 0.5**2:
            ans += 1
        return ans > 0
 
    def _draw(self):
        plt.clf()  # 清除上次画的图
        for node in self.nodes:
            if node.parent is not None:
                plt.plot([node.x, self.nodes[node.parent].x], [
                         node.y, self.nodes[node.parent].y], "-g")
 
 
        plt.plot(self.start.x, self.start.y, "^r")
        plt.plot(self.goal.x, self.goal.y, "^b")
        plt.axis([self.bound.x_min, self.bound.x_max, self.bound.z_min, self.bound.z_max])
        plt.grid(True)
        plt.pause(0.01)

    def planning(self):
        iters = 0
        while iters < self.max_iter:
            iters += 1
            # Random Sampling
            if self.rng.random() > self.sample_rate:
                rnd = self.get_random_node()
            else:
                rnd = [self.goal.x, self.goal.y]

            # Find nearest node
            nearest_node, idx = self.get_nearest_node(self.nodes, rnd)
 
            # 返回弧度制
            theta = math.atan2(rnd[1] - nearest_node.y, rnd[0] - nearest_node.x)
 
            new_x = nearest_node.x + self.step_length * math.cos(theta)
            new_y = nearest_node.y + self.step_length * math.sin(theta)
            new_node = Node([new_x, new_y])
            new_node.parent = idx
 
            if self.collision_check(new_node):
                continue
                
            self.nodes.append(new_node)
 
            dist = ((new_node.x - self.goal.x)**2+(new_node.y - self.goal.y)**2)**0.5
            if dist <= 0.4:
                break
        
        path = [[self.goal.x, self.goal.y]]
        path_nodes = [self.goal]
        last_index = len(self.nodes) - 1
        while self.nodes[last_index].parent is not None:
            node = self.nodes[last_index]
            path.append([node.x, node.y])
            path_nodes.append(node)
            last_index = node.parent
        path.append([self.start.x, self.start.y])
        path_nodes.append(self.start)
        
        del_list = []
        for i in range(1,len(path_nodes)-1):
            del_flag = self.merge_nodes(path_nodes[i-1], path_nodes[i], path_nodes[i+1])
            if del_flag:
                del_list.append(i)
        
        new_path = []
        new_path_nodes = []
        for i in range(len(path)):
            if not(i in del_list):
                new_path.append(path[i])
                new_path_nodes.append(path_nodes[i])

        return new_path, iters<self.max_iter

    def merge_nodes(self, node1, node2, node3):
        del_flag = True
        grid1 = pos_to_grid(node1.x, node2.y, self.bound)
        grid3 = pos_to_grid(node3.x, node3.y, self.bound)
        minx = min(grid1[0], grid3[0])
        maxx = max(grid1[0], grid3[0])
        miny = min(grid1[1], grid3[1])
        maxy = max(grid1[1], grid3[1])
        for x in range(minx, maxx+1):
            for y in range(miny, maxy+1):
                if self.map[x,y] != 0:
                    del_flag = False
        if del_flag:
            node1.parent = node2.parent
        return del_flag
