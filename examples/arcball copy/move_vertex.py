import os.path
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pyglet
import pyglet.gl as GL
import trimesh as tm

import struct
from pygltflib import GLTF2

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
)

import grafica.transformations as tr
from grafica.utils import load_pipeline


def move_vertex(json, vertex_list, time):
    
    

    return