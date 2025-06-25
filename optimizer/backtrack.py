import networkx as nx

from collections import defaultdict
from collections import deque


"""

This optimizer uses backtracking methods to find all possible path from ingress node to egress node within the given graph

"""

def bfs(G, ingress, egress):
    visited = set()     # Keep track of visited nodes
    queue = deque([ingress])  # Initialize the queue with the starting node
    traversal = []

    while queue:  # While there are still nodes to process
        current_node = queue.popleft()  # Dequeue a node from the front of the queue

        if current_node not in visited:  # Check if the node has been visited
            traversal.append(current_node)
            visited.add(current_node)  # Mark the node as visited
            # Enqueue all unvisited neighbors (children) of the current node
            for neighbor in G.neighbors(current_node):
                if neighbor not in visited:
                    queue.append(neighbor)  # Add unvisited neighbors to the queue
                if neighbor == egress:
                    print(traversal)

    return traversal

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