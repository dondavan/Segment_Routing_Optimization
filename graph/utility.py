import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Initialize Graph Edge Parameter
def init_edge_weight(G):
    I_weights = {}

    for edge in G.edges():
        I_weight = np.random.rand()
        I_weights[edge] = I_weight

    nx.set_edge_attributes(G, I_weights, 'I_weight')

# Initialize Graph Node Parameter
def init_node_weight(G):
    T_weights = {}

    for node in G.nodes():
        T_weight = np.random.rand() 
        T_weights[node] = T_weight

    nx.set_node_attributes(G, T_weights, 'T_weight')

def draw_path_in_graph(G,colored_paths,ingress,egress):
    row = 3
    col = (len(colored_paths)-1) //2 +1
    fig, all_axes = plt.subplots(row,col)
    ax = all_axes.flat
    
    _node_size = 400
    """
    Drawing reference 
    """
    # T 
    T_weight = nx.get_node_attributes(G, "T_weight")
    node_labels = T_weight
    for ti in T_weight:
        node_labels[ti] = f't: {T_weight[ti]:.2f}'

    pos = nx.spectral_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=_node_size, ax=ax[0]) # nodes
    nx.draw_networkx_nodes(G, pos, nodelist= [ingress], node_size=_node_size, ax=ax[0],node_color="black") # ingress
    nx.draw_networkx_nodes(G, pos, nodelist= [egress], node_size=_node_size, ax=ax[0],node_color="black") # egress
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif", ax=ax[0], labels=node_labels) # node labels

    e_all = [(u, v) for (u, v) in G.edges()]
    nx.draw_networkx_edges(G, pos, edgelist=e_all, width=2, ax=ax[0]) # edges

    T_weight = nx.get_node_attributes(G, "T_weight")
    node_labels = T_weight
    for ti in T_weight:
        node_labels[ti] = f't: {T_weight[ti]:.2f}'

    #nx.draw_networkx_node_labels(G, pos, node_labels, font_size=10,  ax=ax[0])# edge weight labels
    ax[0].title.set_text('T_weight')
    # I
    pos = nx.spectral_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=_node_size, ax=ax[1]) # nodes
    nx.draw_networkx_nodes(G, pos, nodelist= [ingress], node_size=_node_size, ax=ax[1],node_color="black") # ingress
    nx.draw_networkx_nodes(G, pos, nodelist= [egress], node_size=_node_size, ax=ax[1],node_color="black") # egress
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif", ax=ax[1]) # node labels

    e_all = [(u, v) for (u, v) in G.edges()]
    nx.draw_networkx_edges(G, pos, edgelist=e_all, width=2, ax=ax[1]) # edges

    I_weight = nx.get_edge_attributes(G, "I_weight")
    edge_labels = I_weight
    for ti in I_weight:
        edge_labels[ti] = f'i: {I_weight[ti]:.2f}'

    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10,  ax=ax[1])# edge weight labels
    ax[1].title.set_text('I_weight')




    """
    Drawing colored path
    """
    for i_colored_paths in range(0,len(colored_paths)):
        value = list(colored_paths.items())[i_colored_paths][1]
        key = list(colored_paths.keys())[i_colored_paths]

        pos = nx.spectral_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=_node_size, ax=ax[i_colored_paths+row-1]) # nodes
        nx.draw_networkx_nodes(G, pos, nodelist= [ingress], node_size=_node_size, ax=ax[i_colored_paths+row-1],node_color="black") # ingress
        nx.draw_networkx_nodes(G, pos, nodelist= [egress], node_size=_node_size, ax=ax[i_colored_paths+row-1],node_color="black") # egress
        nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif", ax=ax[i_colored_paths+row-1]) # node labels

        # Draw the graph using plt
        e_all = [(u, v) for (u, v) in G.edges()]
        #nx.draw_networkx_edges(G, pos, edgelist=e_all, width=2, ax=ax[i_colored_paths+row-1],alpha=1, arrows=False) # edges

        if(len(value)>0):
            e_colored = [(colored_paths[key][0][i-1],colored_paths[key][0][i]) for i in range(1,len(colored_paths[key][0]))]
            nx.draw_networkx_edges(G, pos, edgelist=e_colored, width=2, alpha=1, edge_color=key, ax=ax[i_colored_paths+row-1]) # colored 
        ax[i_colored_paths+row-1].title.set_text(key)
        # edge weight labels
        #edge_labels = nx.get_edge_attributes(G, "T_weight")
        #x.draw_networkx_edge_labels(G, pos, edge_labels)

    """
    Drawing combined path
    """
    print(colored_paths)
    for i_colored_paths in range(0,len(colored_paths)):
        value = list(colored_paths.items())[i_colored_paths][1]
        key = list(colored_paths.keys())[i_colored_paths]

        pos = nx.spectral_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=_node_size, ax=ax[len(colored_paths)+row-1]) # nodes
        nx.draw_networkx_nodes(G, pos, nodelist= [ingress], node_size=_node_size, ax=ax[len(colored_paths)+row-1],node_color="black") # ingress
        nx.draw_networkx_nodes(G, pos, nodelist= [egress], node_size=_node_size, ax=ax[len(colored_paths)+row-1],node_color="black") # egress
        nx.draw_networkx_labels(G, pos, font_size=5, font_family="sans-serif", ax=ax[len(colored_paths)+row-1]) # node labels

        # Draw the graph using plt
        e_all = [(u, v) for (u, v) in G.edges()]
        #nx.draw_networkx_edges(G, pos, edgelist=e_all, width=2, ax=ax[len(colored_paths)+row-1]) # edges

        rad = 0.1
        if(len(value)>0):
            e_colored = [(colored_paths[key][0][i-1],colored_paths[key][0][i]) for i in range(1,len(colored_paths[key][0]))]
            nx.draw_networkx_edges(G, pos, edgelist=e_colored, width=2, alpha=1, edge_color=key, ax=ax[len(colored_paths)+row-1], 
                       connectionstyle=f'arc3, rad = {rad*i_colored_paths}') # colored 
        ax[len(colored_paths)+row-1].title.set_text("colored")

    for a in ax:
        a.margins(0.10)
    fig.tight_layout()
    plt.show()
