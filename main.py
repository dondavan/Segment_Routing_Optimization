import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum

from optimizer import backtrack,refine
from graph import grid,utility


# Hyper Parameters
dim=(10, 10)          # Graph Dimension, example 5*5 2d graph
ingress=(0,0)       # Ingress node, (x,y)
egress=(9,9)        # Egress node, (x,y)

# Constrain
colors = ['red','blue','yellow']
T = [0.1,0.3,0.5]
I = [0.1,0.3,0.5]


# Graph attributes
NODE_ATTRIBUTES = ['Integrity', 'Resilience', 'Confidentiality']
EDGE_ATTRIBUTES = ['Latency', 'Bandwidth', 'HopCount']

# Optimization Objectives
ATTRIBUTE_VALUES = Enum('ATTRIBUTE_VALUES', [('VL', 1), ('L', 2), ('M', 3), ('H', 4), ('VH', 5)])
POLICY = {
    'Integrity':['M','H','VH'],
    'Bandwidth': ['M','H','VH'],
    'Confidentiality':['VH'],
    'HopCount':['VL','L','M','H','VH'],
    'Latency':['L','M','H','VH'],
    
    'Resilience':['VL','L','M','H','VH'],
}

# Load Graph and Init Egde Weight
G = grid.get_graph(dim=dim,ingress=ingress,egress=egress)

utility.init_edge_weight(G,EDGE_ATTRIBUTES, POLICY)
utility.init_node_weight(G,NODE_ATTRIBUTES, POLICY)


# Optimization
#paths = backtrack.get_path(G,ingress,egress)

"""
colored_paths = {}
# Constrain is >= at T,I at the same time
for i in range(0,len(colors)):
    constrain = {colors[i]: {"T_weight":T[i], "I_weight":I[i]}}  
    refined_paths = refine.apply_constrain_accumlate(G,paths,constrain)
    colored_paths[colors[i]] = refined_paths

utility.draw_path_in_graph(G,colored_paths,ingress,egress)
"""

utility.draw_graph(G,NODE_ATTRIBUTES, EDGE_ATTRIBUTES, ingress,egress)