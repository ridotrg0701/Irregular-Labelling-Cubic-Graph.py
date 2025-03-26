import networkx as nx
import matplotlib.pyplot as plt

def divide_equal_parts(dividend, divisor):
    quotient = dividend // divisor
    remainder = dividend % divisor
    equal_parts = [quotient] * (divisor - remainder) + [quotient + 1] * remainder
    equal_parts.sort(reverse=True)
    return equal_parts

def generate_cycle_graph(n):
    # Generate cycle A
    cycle_A = nx.cycle_graph(n)
    
    # Generate cycle B
    cycle_B = nx.cycle_graph(n)
    
    # Rename nodes of cycle B to match with cycle A
    mapping = {node: node + n for node in cycle_B.nodes()}
    cycle_B = nx.relabel_nodes(cycle_B, mapping)
    
    # Combine cycle A and cycle B
    combined_graph = nx.compose(cycle_A, cycle_B)
    
    # Add edges between corresponding nodes in cycle A and cycle B
    for i in range(n):
        combined_graph.add_edge(i, i + n, weight=0)  # Ensure edges have weights (0 by default)
    
    return combined_graph

def bfs_node_labels(graph, start_node):
    # Perform Breadth-First Search
    bfs_order = list(nx.bfs_edges(graph, start_node))
    
    # Initialize labels dictionary
    labels = {start_node: 0}
    current_label = 1
    
    # Assign labels to nodes based on BFS traversal order
    for edge in bfs_order:
        for node in edge:
            if node not in labels:
                labels[node] = current_label
                current_label += 1
    
    # Add 4 to each label
    labels = {node: label + 4 for node, label in labels.items()}
    
    return labels

def assign_weights(graph, start_node):
    # Perform BFS node labeling
    node_labels = bfs_node_labels(graph, start_node)
    
    # Print node labels for debugging
    #print("Node Labels:", node_labels)
    
    # Set weight for all nodes and edges to 0
    for node in graph.nodes():
        graph.nodes[node]['weight'] = 0
    for edge in graph.edges():
        graph.edges[edge]['weight'] = 0
    
    # Initialize dictionaries to store node and edge weights
    node_weights = {}
    edge_weights = {}
    
    # Iterate through nodes
    for node, label in node_labels.items():
        # Find adjacent edges with zero weight and non-zero weight
        zero_edge = [(u, v) for u, v in graph.edges(node) if graph[u][v].get('weight', 0) == 0]
        non_zero_edge = [(u, v) for u, v in graph.edges(node) if graph[u][v].get('weight', 0) != 0]
        
        # Sort zero edges based on the highest vertex label
        zero_edge.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate initial weight for the node
        weight_start = label - sum(graph[u][v].get('weight', 0) for u, v in non_zero_edge)
        
        '''
        # Print debugging information
        print("Node:", node)
        print("Label:", label)
        print("Non-zero edges:", non_zero_edge)
        print("Initial Weight Start:", weight_start)
        '''
        
        # Count the number of zero weight edges
        count_edge_zero = len(zero_edge)
        
        '''
        # Print debugging information
        print("Zero Edges:", zero_edge)
        print("Count of Zero Edges:", count_edge_zero)
        '''
        
        # Divide the initial weight among zero weight edges
        weight_parts = divide_equal_parts(weight_start, count_edge_zero + 1)
        
        '''
        # Print debugging information
        print("Weight Parts:", weight_parts)
        '''
        
        # Assign weights to zero weight edges
        for edge in zero_edge:
            graph.edges[edge]['weight'] = weight_parts[0]
            weight_parts.pop(0)
        
        # Assign the remaining weight to the node
        graph.nodes[node]['weight'] = weight_parts[0]
        
        # Store weights in dictionaries
        node_weights[node] = graph.nodes[node]['weight']
        for edge in zero_edge:
            edge_weights[edge] = graph.edges[edge]['weight']
        
    return node_weights, edge_weights


def visualize_graph(graph, node_labels, node_weights, edge_weights, seed=42):
    # Remove default labels
    for node in graph.nodes():
        graph.nodes[node].pop('label', None)

    # Create label dictionary with format (node_label, node_weight)
    labels = {node: f"({node_labels[node]}, {node_weights[node]})" for node in graph.nodes()}

    # Set random seed for layout consistency
    pos = nx.spring_layout(graph, seed=seed)

    # Draw graph
    nx.draw(graph, pos, with_labels=False, node_color='skyblue', node_size=1000, edge_color='gray')
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, font_color='black', font_weight='bold')

    # Draw edge weights in green
    for edge, weight in edge_weights.items():
        nx.draw_networkx_edge_labels(graph, pos, edge_labels={(edge[0], edge[1]): weight}, font_color='green', font_size=10)

    # Show the graph
    plt.axis('off')
    plt.show()

def graph_info(node_labels, node_weights, edge_weights):
    print("(Node Label, Node Weight):")
    for node, label in node_labels.items():
        print(f"{label}: {node_weights[node]}")

    print("\nEdge Weight:")
    for edge, weight in edge_weights.items():
        print(f"{edge}: {weight}")

    # Calculate Total Vertex Strength (TVS)
    max_weight = max(max(node_weights.values()), max(edge_weights.values()))
    print(f"\nTVS: {max_weight}")


# Generate graph with n=4 and assign weights
n = int(input("Number of nodes on the cubic base: "))
graph = generate_cycle_graph(n)
start_node = 0
node_labels = bfs_node_labels(graph, start_node)
node_weights, edge_weights = assign_weights(graph, start_node)

# Graph info
graph_info(node_labels, node_weights, edge_weights)
              
# Visualize the graph with specified labels and weights and set seed
visualize_graph(graph, node_labels, node_weights, edge_weights, seed=42)







