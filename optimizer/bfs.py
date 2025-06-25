from collections import deque

"""

This optimizer uses bfs to find all possible path from ingress node to egress node within the given graph

"""

def bfs(G, ingress, egress):
    visited = set()     # Keep track of visited nodes
    logged = set()    # Keep tranck of logged nodes
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

def get_path(G,ingress,egress):
    Paths = []
    path = bfs(G=G,ingress=ingress,egress=egress)
    print(path)
    return Paths