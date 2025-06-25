import networkx as nx
import numpy as np

# Initialize Graph with I and T
def init_edge_weight(G):
    I_weights = {}
    T_weights = {}
    for edge in G.edges():
        I_weight = np.random.rand()
        I_weights[edge] = I_weight

        T_weight = np.random.rand()
        T_weights[edge] = T_weight
    
    nx.set_edge_attributes(G, T_weights, 'T_weight')
    nx.set_edge_attributes(G, I_weights, 'I_weight')