# app.py - The full Evolution Strategies Optimizer

import numpy as np
import matplotlib.pyplot as plt
import random
import copy

# --- Component 1: The TSP Solver (from our previous step) ---
# We'll use this inside our fitness calculation.
def solve_tsp_nearest_neighbor(dist_matrix):
    num_items = dist_matrix.shape[0]
    start_node_idx = 0 # Assume DOCK is always index 0
    
    current_node_idx = start_node_idx
    path_indices = [current_node_idx]
    unvisited_indices = list(range(num_items))
    unvisited_indices.remove(start_node_idx)
    
    total_distance = 0
    
    while unvisited_indices:
        # Find the closest unvisited neighbor from the current node
        distances_from_current = dist_matrix[current_node_idx, unvisited_indices]
        nearest_neighbor_local_idx = np.argmin(distances_from_current)
        nearest_neighbor_idx = unvisited_indices[nearest_neighbor_local_idx]
        
        min_distance = dist_matrix[current_node_idx, nearest_neighbor_idx]
        
        total_distance += min_distance
        current_node_idx = nearest_neighbor_idx
        path_indices.append(current_node_idx)
        unvisited_indices.remove(current_node_idx)
        
    total_distance += dist_matrix[current_node_idx, start_node_idx]
    return total_distance

# --- Component 2: The Evolution Strategies Algorithm ---

def initialize_population(product_ids, width, height, population_size):
    """Creates a population of random warehouse layouts."""
    population = []
    all_coords = [(y, x) for y in range(height) for x in range(width)]
    
    for _ in range(population_size):
        random.shuffle(all_coords)
        # A layout is a dictionary mapping each product to a (y,x) coordinate
        layout = {product_id: coord for product_id, coord in zip(product_ids, all_coords)}
        population.append(layout)
        
    return population

def calculate_fitness(layout, product_ids, num_test_orders=50, items_per_order=10):
    """
    Calculates the fitness of a single layout. Lower fitness is better.
    Fitness = Average TSP distance over many random orders.
    """
    total_tour_distance = 0
    
    for _ in range(num_test_orders):
        # 1. Create a random pick list for this test
        pick_list_products = ['DOCK'] + random.sample(product_ids, items_per_order)
        
        # 2. Get the locations for these specific products from the input layout
        item_locations = {prod: layout[prod] for prod in pick_list_products}
        
        # 3. Calculate the distance matrix for this pick list
        item_names = list(item_locations.keys())
        num_items = len(item_names)
        dist_matrix = np.zeros((num_items, num_items))
        for i in range(num_items):
            for j in range(num_items):
                loc1 = item_locations[item_names[i]]
                loc2 = item_locations[item_names[j]]
                dist_matrix[i, j] = abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1]) # Manhattan Distance
        
        # 4. Solve the TSP for this pick list and add to total
        tour_distance = solve_tsp_nearest_neighbor(dist_matrix)
        total_tour_distance += tour_distance
        
    return total_tour_distance / num_test_orders

def mutate(parent_layout):
    """
    Creates a new 'child' layout by making a small change to the parent.
    The mutation strategy here is to swap the locations of two random products.
    """
    child_layout = copy.deepcopy(parent_layout)
    # Select two random products to swap
    prod1, prod2 = random.sample(list(child_layout.keys()), 2)
    
    # Swap their locations
    loc1 = child_layout[prod1]
    loc2 = child_layout[prod2]
    child_layout[prod1] = loc2
    child_layout[prod2] = loc1
    
    return child_layout

def visualize_best_layout(layout, width, height, product_sales_map):
    """Visualizes the final best layout as a heatmap."""
    # Create a grid and populate it with sales data for coloring
    grid = np.zeros((height, width))
    for product, coord in layout.items():
        if product != 'DOCK':
            grid[coord[0], coord[1]] = product_sales_map[product]
            
    plt.figure(figsize=(8, 6))
    plt.imshow(grid, cmap='hot', interpolation='nearest')
    plt.title("Final Evolved Warehouse Layout", fontsize=16)
    plt.xlabel("Aisle")
    plt.ylabel("Shelf")
    plt.colorbar(label="Product Sales Volume (Hot = High Sales)")
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # --- Hyperparameters for the Optimizer ---
    WAREHOUSE_WIDTH = 10
    WAREHOUSE_HEIGHT = 10
    NUM_PRODUCTS = 50
    
    POPULATION_SIZE = 20  # How many layouts to test in each generation
    NUM_GENERATIONS = 50  # How many generations to evolve for
    NUM_PARENTS = 5       # How many of the "best" layouts survive to the next generation
    
    # --- Setup ---
    # Create product IDs and a mock sales map (for the final visualization)
    product_ids_only = [f"SKU{1000+i}" for i in range(NUM_PRODUCTS)]
    product_sales = {pid: (NUM_PRODUCTS - i) * 10 for i, pid in enumerate(product_ids_only)} # Make first SKUs more popular
    all_products_with_dock = ['DOCK'] + product_ids_only
    
    # --- The Evolution Loop ---
    print("Initializing a random population of layouts...")
    population = initialize_population(all_products_with_dock, WAREHOUSE_WIDTH, WAREHOUSE_HEIGHT, POPULATION_SIZE)
    
    best_fitness_history = []
    
    print(f"Starting evolution for {NUM_GENERATIONS} generations...")
    for gen in range(NUM_GENERATIONS):
        # 1. Evaluate the fitness of every layout in the population
        fitness_scores = [calculate_fitness(layout, product_ids_only) for layout in population]
        
        # 2. Select the best parents
        sorted_indices = np.argsort(fitness_scores)
        parents = [population[i] for i in sorted_indices[:NUM_PARENTS]]
        best_fitness = fitness_scores[sorted_indices[0]]
        best_fitness_history.append(best_fitness)
        
        print(f"Generation {gen+1}/{NUM_GENERATIONS} - Best Avg. Distance: {best_fitness:.2f}")
        
        # 3. Create the next generation
        next_population = parents # The best parents automatically survive
        num_children = POPULATION_SIZE - NUM_PARENTS
        
        for i in range(num_children):
            # Pick a random parent to mutate
            parent = random.choice(parents)
            child = mutate(parent)
            next_population.append(child)
            
        population = next_population

    # --- Final Results ---
    print("\nEvolution complete!")
    final_fitness_scores = [calculate_fitness(layout, product_ids_only) for layout in population]
    best_layout_index = np.argmin(final_fitness_scores)
    best_layout = population[best_layout_index]
    
    # Plot the history of improvement
    plt.figure(figsize=(10, 5))
    plt.plot(best_fitness_history)
    plt.title("Optimization Progress Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Best Average Travel Distance")
    plt.grid(True)
    plt.show()
    
    # Visualize the best layout found
    visualize_best_layout(best_layout, WAREHOUSE_WIDTH, WAREHOUSE_HEIGHT, product_sales)
