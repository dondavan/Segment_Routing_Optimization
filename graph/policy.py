from enum import Enum

class graph_w_attributes:
    node_attributes = []
    edge_attributes = []
    
    
    def __init__(self,node_attributes, edge_attributes):
        self.node_attributes = node_attributes
        self.edge_attributes = edge_attributes