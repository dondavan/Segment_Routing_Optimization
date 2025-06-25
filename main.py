import networkx as nx
import matplotlib.pyplot as plt

from optimizer import backtrack
from graph import grid,utility


# Hyper Parameters
dim=(3, 3)          # Graph Dimension, example 5*5 2d graph
ingress=(0,0)       # Ingress node, (x,y)
egress=(2,2)        # Egress node, (x,y)

# Constrain
colors = ['red','blue','yellow','black']
T = [0.2,0.4,0.6,0.8]
I = [0.2,0.4,0.6,0.8]


# Load Graph and Init Egde Weight
G = grid.get_graph(dim=dim,ingress=ingress,egress=egress)
utility.init_edge_weight(G)

print(G.edges())

# Optimization
Paths = backtrack.get_path(G,ingress,egress)


# Draw the graph using plt
elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['T_weight'] > 0.5]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['T_weight'] <= 0.5]

pos = nx.spectral_layout(G)  # positions for all nodes - seed for reproducibility

# nodes
nx.draw_networkx_nodes(G, pos, node_size=700)
# edges
nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
nx.draw_networkx_edges(
    G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
)

# node labels
nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
# edge weight labels
edge_labels = nx.get_edge_attributes(G, "T_weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)

ax = plt.gca()
ax.margins(0.08)
plt.axis("off")
plt.tight_layout()
#plt.show()