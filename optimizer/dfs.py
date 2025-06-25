import networkx as nx

from collections import defaultdict

"""

This optimizer uses dfs to find all possible path from ingress node to egress node within the given graph

"""

def dfs(current_node, egress, allPaths, currPath, currPath_visited, G):
    if current_node == egress:
        allPaths.append([x for x in currPath])
        return
    
    for adj_Node in G.neighbors(current_node):
        if not currPath_visited[adj_Node]:

            currPath_visited[adj_Node] = True
            currPath.append(adj_Node)

            dfs(adj_Node, egress, allPaths, currPath, currPath_visited, G)
            
            # Backtrack
            currPath.pop()
            currPath_visited[adj_Node] = False

def get_path(G,ingress,egress):
    currPath_visited = defaultdict(bool)
    currPath_visited[ingress] = True
    
    # To keep track of paths
    allPaths = []
    currPath = [ingress]
    
    # Call function
    dfs(ingress, egress, allPaths, currPath, currPath_visited, G)
    return allPaths