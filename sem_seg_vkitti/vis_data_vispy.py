# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 20:28:39 2018

@author: Isaac
"""
# -*- coding: utf-8 -*-
# vispy: gallery 10
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

""" Demonstrates use of visual.Markers to create a point cloud with a
standard turntable camera to fly around with and a centered 3D Axis.
"""

import numpy as np
import vispy.scene
from vispy.scene import visuals
import os

# set canvas
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

# data sample
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
point_cloud = np.load('{}/data/vkitti3d_dataset_v1.0/03/0002_00031.npy'.format(ROOT_DIR))
#print(point_cloud[0, :])
data1 = point_cloud[:,:6]
label1 = point_cloud[:,-1]
print(data1.shape, label1.shape)
print(np.min(label1))
print(np.max(label1))

# create scatter object and fill in the data
scatter = visuals.Markers()
rgb_codes_1 = [[255,0,0],
            [255,255,0],
            [255,0,255],
            [0,0,255],
            [0,255,0],
            [0,255,255],
            [100,100,100],
            [255,30,50],
            [25,180,60],
            [5,220,100],
            [25,100,200],
            [105,60,90],
            [150,0,190]]

rgb_codes = [[200, 90, 0],
            [0, 128, 50],
            [0, 220, 0],
            [255, 0, 0],
            [100, 100, 100],
            [200, 200, 200],
            [255, 0, 255],
            [255, 255, 0],
            [128, 0, 255],
            [255, 200, 150],
            [0, 128, 255],
            [0, 200, 255],
            [255, 128, 0],
            [0, 0, 0]]
'''
0	Terrain	[200, 90, 0]	brown
1	Tree	[0, 128, 50]	dark green
2	Vegetation	[0, 220, 0]	bright green
3	Building	[255, 0, 0]	red
4	Road	[100, 100, 100]	dark gray
5	GuardRail	[200, 200, 200]	bright gray
6	TrafficSign	[255, 0, 255]	pink
7	TrafficLight	[255, 255, 0]	yellow
8	Pole	[128, 0, 255]	violet
9	Misc	[255, 200, 150]	skin
10	Truck	[0, 128, 255]	dark blue
11	Car	[0, 200, 255]	bright blue
12	Van	[255, 128, 0]	orange
13	Don't care	[0, 0, 0]	black
'''
color = np.zeros((label1.shape[0], 3))

for i in range(label1.shape[0]):
    # Label color
    #color[i, :] = [code/255 for code in rgb_codes[int(label1[i])]]
    # Data color
    color[i, :] = [code / 255 for code in data1[i, 3:]]


scatter.set_data(data1[:,:3], edge_color=None, face_color=color, size=5)

view.add(scatter)
view.camera = 'turntable'  # or try 'arcball'

# add a colored 3D axis for orientation
axis = visuals.XYZAxis(parent=view.scene)

if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        vispy.app.run()
