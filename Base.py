# app.py - Component 1: TSP Solver for a single pick list

import numpy as np

def create_warehouse_and_items(width, height, num_items_in_list):
    """
    Creates a simple warehouse grid and a random list of item locations to be picked.
    The first item is always the shipping dock at (0, 0).
    """
    # Create a dictionary of all possible item locations (their coordinates)
    # For example: {'item_0': (0, 1), 'item_1': (5, 2), ...}
    all_locations = {f"item_{i}": (np.random.randint(0, height), np.random.randint(0, width)) 
                     for i in range(width * height)}
    
    # Create a random pick list
    pick_list_keys = np.random.choice(list(all_locations.keys()), size=num_items_in_list, replace=False)
    
    # Get the coordinates for the items in our pick list
    item_locations = {key: all_locations[key] for key in pick_list_keys}
    
    # Add the shipping dock, which is always the start and end point
    item_locations['DOCK'] = (0, 0)
    
    return item_locations

def calculate_distance_matrix(item_locations):
    """
    Calculates the Manhattan distance between every pair of items in the pick list.
    Returns a distance matrix and a mapping from item name to index.
    """
    item_names = list(item_locations.keys())
    num_items = len(item_names)
    dist_matrix = np.zeros((num_items, num_items))
    
    # Create a map from item name to its index in the matrix (e.g., 'DOCK' -> 0)
    item_to_idx = {name: i for i, name in enumerate(item_names)}
    
    for i in range(num_items):
        for j in range(num_items):
            loc1 = item_locations[item_names[i]]
            loc2 = item_locations[item_names[j]]
            # Manhattan Distance = |y1 - y2| + |x1 - x2|
            dist_matrix[i, j] = abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
            
    return dist_matrix, item_to_idx, item_names

def solve_tsp_nearest_neighbor(dist_matrix, item_to_idx, item_names):
    start_node_idx = item_to_idx['DOCK']
    num_items = len(item_names)
    
    current_node_idx = start_node_idx
    path_indices = [current_node_idx]
    unvisited_indices = list(range(num_items))
    unvisited_indices.remove(start_node_idx)
    
    total_distance = 0
    
    while unvisited_indices:
        nearest_neighbor_idx = -1
        min_distance = float('inf')
        
        # Find the closest unvisited neighbor
        for neighbor_idx in unvisited_indices:
            if dist_matrix[current_node_idx, neighbor_idx] < min_distance:
                min_distance = dist_matrix[current_node_idx, neighbor_idx]
                nearest_neighbor_idx = neighbor_idx
        
        # Move to the nearest neighbor
        total_distance += min_distance
        current_node_idx = nearest_neighbor_idx
        path_indices.append(current_node_idx)
        unvisited_indices.remove(current_node_idx)
        
    # Add the distance to return to the dock
    total_distance += dist_matrix[current_node_idx, start_node_idx]
    path_indices.append(start_node_idx)
    
    # Convert path indices back to item names for readability
    path_names = [item_names[i] for i in path_indices]
    
    return path_names, total_distance

# --- Main Execution ---
if __name__ == "__main__":
    # --- Parameters ---
    WAREHOUSE_WIDTH = 20
    WAREHOUSE_HEIGHT = 20
    ITEMS_IN_ORDER = 10
    
    # 1. Create a sample warehouse and a random customer order (pick list)
    item_locations = create_warehouse_and_items(WAREHOUSE_WIDTH, WAREHOUSE_HEIGHT, ITEMS_IN_ORDER)
    print(f"A random order was generated for {ITEMS_IN_ORDER} items plus the DOCK.")
    # print("Item Locations:", item_locations)
    
    # 2. Calculate the distance matrix for this specific order
    dist_matrix, item_to_idx, item_names = calculate_distance_matrix(item_locations)
    
    # 3. Solve for the best picking route for THIS order
    best_path, total_distance = solve_tsp_nearest_neighbor(dist_matrix, item_to_idx, item_names)
    
    print("\n--- TSP Solver Results ---")
    print(f"Optimized Picking Route: {' -> '.join(best_path)}")
    print(f"Total Travel Distance for this Route: {total_distance:.2f} units")
