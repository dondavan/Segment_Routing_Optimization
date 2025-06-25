import networkx as nx

def get_graph(dim=(5,5),ingress=(0,0),egress=(4,4)):
    G = nx.grid_graph(dim)
    nodes = list(G.nodes())

    if nodes.count(ingress) == 0:
        raise RuntimeError('Ingress node can not fit in graph')
    if nodes.count(egress) == 0:
        raise RuntimeError('Egress node can not fit in graph')
    DG = nx.DiGraph(G)
    return DG