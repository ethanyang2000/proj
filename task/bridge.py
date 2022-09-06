import os
from turtle import position

import numpy as np
import cv2
import pyastar2d as pyastar
import random
import time
import math
from icecream import ic
from utils import a_star_search


CELL_SIZE = 0.49
ANGLE = 15

def pos2map(x, z, _scene_bounds):
    i = int(round((x - _scene_bounds["x_min"]) / CELL_SIZE))
    j = int(round((z - _scene_bounds["z_min"]) / CELL_SIZE))
    return i, j

class bridge:
    def __init__(self):
        self.map_size = (55,22)#(120, 60)
        self._scene_bounds = {
            "x_min": -15,
            "x_max": 15,
            "z_min": -7.5,
            "z_max": 7.5
        }
        self._reset()
        self.obs = None
        
    
    def pos2map(self, x, z):
        i = int(round((x - self._scene_bounds.x_min) / CELL_SIZE))
        j = int(round((z - self._scene_bounds.z_min) / CELL_SIZE))
        return i, j
        
    def map2pos(self, i, j):
        x = i * CELL_SIZE + self._scene_bounds.x_min
        z = j * CELL_SIZE + self._scene_bounds.z_min
        return x, z 
    
    
    def dep2map(self, obs):
        self.obs = obs
        local_occupancy_map = np.zeros_like(self.occupancy_map, np.int32)
        local_known_map = np.zeros_like(self.occupancy_map, np.int32)
        depth = self.obs['depth']
        
        #camera info
        FOV = self.obs['FOV']
        W, H = depth.shape
        cx = W / 2.
        cy = H / 2.
        fx = cx / np.tan(math.radians(FOV / 2.))
        fy = cy / np.tan(math.radians(FOV / 2.))
        
        #Ego
        x_index = np.linspace(0, W - 1, W)
        y_index = np.linspace(0, H - 1, H)
        xx, yy = np.meshgrid(x_index, y_index)
        xx = (xx - cx) / fx * depth
        yy = (yy - cy) / fy * depth
        
        pc = np.stack((xx, yy, depth, np.ones((xx.shape[0], xx.shape[1]))))  
        
        pc = pc.reshape(4, -1)
        
        E = self.obs['camera_matrix']
        inv_E = np.linalg.inv(np.array(E).reshape((4, 4)))
        rot = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
        inv_E = np.dot(inv_E, rot)
        
        rpc = np.dot(inv_E, pc).reshape(4, W, H)
        
        rpc = rpc.reshape(4, -1)
        X = np.rint((rpc[0, :] - self._scene_bounds["x_min"]) / CELL_SIZE)
        X = np.maximum(X, 0)
        X = np.minimum(X, self.map_size[0] - 1)
        Z = np.rint((rpc[2, :] - self._scene_bounds["z_min"]) / CELL_SIZE)
        Z = np.maximum(Z, 0)
        Z = np.minimum(Z, self.map_size[1] - 1)
        depth = depth.reshape(-1)
        index = np.where((depth > 0.7) & (depth < 99) & (rpc[1, :] < 1.0))
        XX = X[index].copy()
        ZZ = Z[index].copy()
        XX = XX.astype(np.int32)
        ZZ = ZZ.astype(np.int32)
        
        index = np.where((depth > 1.3) & (depth < 99) & (rpc[1, :] < -0.5))
        XX = X[index].copy()
        ZZ = Z[index].copy()
        XX = XX.astype(np.int32)
        ZZ = ZZ.astype(np.int32)
        self.occupancy_map[XX, ZZ] = 0
        
        index = np.where((depth > 0.7) & (depth < 95) & (rpc[1, :] > 0.0) & (rpc[1, :] < 1.3))
        XX = X[index]
        ZZ = Z[index]
        XX = XX.astype(np.int32)
        ZZ = ZZ.astype(np.int32)
        self.occupancy_map[XX, ZZ] = 1
        return local_known_map
        np.savetxt('map.txt', self.occupancy_map,fmt='%d')
            
    def l2_distance(self, st, g):
        return ((st[0] - g[0]) ** 2 + (st[1] - g[1]) ** 2) ** 0.5
    
    def get_angle(self, forward, origin, position):        
        p0 = np.array([origin[0], origin[2]])
        p1 = np.array([position[0], position[2]])
        d = p1 - p0
        d = d / np.linalg.norm(d)
        f = np.array([forward[0], forward[2]])

        dot = f[0] * d[0] + f[1] * d[1]
        det = f[0] * d[1] - f[1] * d[0]
        angle = np.arctan2(det, dot)
        angle = np.rad2deg(angle)
        return angle

    
    def conv2d(self, map, kernel=3):
        from scipy.signal import convolve2d
        conv = np.ones((kernel, kernel))
        return convolve2d(map, conv, mode='same', boundary='fill')
    
    def find_shortest_path(self, st, goal, map = None):
    
        st_x, _, st_z = st
        g_x, g_z = goal
        st_i, st_j = self.pos2map(st_x, st_z)
        g_i, g_j = self.pos2map(g_x, g_z)
        dist_map = np.ones_like(map, dtype=np.float32)
        super_map1 = self.conv2d(map, kernel=5)
        dist_map[super_map1 > 0] = 10
        super_map2 = self.conv2d(map)
        dist_map[super_map2 > 0] = 1000
        dist_map[map > 0] = 100000
        np.savetxt('dis.txt', dist_map, fmt='%d')
        dist_map = map
        path = a_star_search(dist_map, (st_i, st_j), (g_i, g_j), None, True)
        
        return path
             
    def _reset(self):
        #self.map_size = self.info['map_size']
        self.W = self.map_size[0]
        self.H = self.map_size[1]
        #self._scene_bounds = self.info['_scene_bounds']
        
        #0: free, 1: occupied
        self.occupancy_map = np.zeros(self.map_size, np.int32)
        self.known_map = np.zeros(self.map_size, np.int32)
        #0: unknown, 1: known        
    
    def store_map(self, obs, map, scene_bound):
        self.occupancy_map = map
        self.obs = obs
        self._scene_bounds = scene_bound

    def nav(self, goal):
        
        if self.obs is None:
            return 0
        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]

        #local_known_map = self.dep2map()
        #self.known_map = np.maximum(self.known_map, local_known_map)
            
        path = self.find_shortest_path(self.position, goal, \
                                        self.occupancy_map)
        i, j = (path)[min(5, len(path) - 1)]
        x, z = self.map2pos(i, j)
        self.local_goal = [x, z]
        angle = self.get_angle(forward=np.array(self.forward),
                            origin=np.array(self.position),
                            position=np.array([self.local_goal[0], 0, self.local_goal[1]]))


        if np.abs(angle) < ANGLE:
            action = 0      
        elif angle > 0:
            action = 1      
        else:
            action = 2

        return action
    
