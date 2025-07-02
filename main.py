import networkx as nx
import matplotlib.pyplot as plt

from optimizer import backtrack,refine
from graph import grid,utility


# Hyper Parameters
dim=(3, 3)          # Graph Dimension, example 5*5 2d graph
ingress=(0,0)       # Ingress node, (x,y)
egress=(2,2)        # Egress node, (x,y)

# Constrain
colors = ['red','blue','yellow']
T = [0.1,0.3,0.5]
I = [0.1,0.3,0.5]


# Load Graph and Init Egde Weight
G = grid.get_graph(dim=dim,ingress=ingress,egress=egress)

utility.init_edge_weight(G)
utility.init_node_weight(G)


# Optimization
paths = backtrack.get_path(G,ingress,egress)

colored_paths = {}
# Constrain is >= at T,I at the same time
for i in range(0,len(colors)):
    constrain = {colors[i]: {"T_weight":T[i], "I_weight":I[i]}}  
    refined_paths = refine.apply_constrain_accumlate(G,paths,constrain)
    colored_paths[colors[i]] = refined_paths

utility.draw_path_in_graph(G,colored_paths)