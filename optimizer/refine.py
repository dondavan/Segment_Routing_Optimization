import networkx as nx

def apply_constrain_min(G, paths,constrain):
    qualified_path = []

    T_weights =nx.get_edge_attributes(G,'T_weight')
    I_weights =nx.get_edge_attributes(G,'I_weight')

    for path in paths:
        qualified = True
        
        # Disqualify path
        for i in range(1,len(path)):
            if T_weights[(path[i-1],path[i])] < list(constrain.values())[0]['T_weight']:
                qualified = False
                break
            if I_weights[(path[i-1],path[i])] < list(constrain.values())[0]['I_weight']:
                qualified = False
                break

        if qualified:
            qualified_path.append(path)

    return(qualified_path)

def apply_constrain_accumlate(G, paths,constrain):
    qualified_path = []

    T_weights =nx.get_node_attributes(G,'T_weight')
    I_weights =nx.get_edge_attributes(G,'I_weight')

    for path in paths:
        qualified = True
        T_sum = 0
        I_sum = 0

        # Accumlate
        for i in range(0,len(path)):
            T_sum = T_sum + T_weights[path[i]]
        for i in range(1,len(path)):
            I_sum = I_sum + I_weights[(path[i-1],path[i])]

        T_avg = T_sum/len(path)
        I_avg = T_sum/(len(path)-1)
        # Disqualify path
        if T_avg < list(constrain.values())[0]['T_weight']:
            qualified = False
        if I_avg < list(constrain.values())[0]['I_weight']:
            qualified = False
            
        if qualified:
            qualified_path.append(path)


    return(qualified_path)