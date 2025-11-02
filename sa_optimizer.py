# sa_optimizer_v2.py
# This is the upgraded SA algorithm that uses weighted data.

import numpy as np
import pandas as pd
import random
import time
import math
import copy

# --- Core Parameters ---
WAREHOUSE_WIDTH = 10
WAREHOUSE_HEIGHT = 10
NUM_PRODUCTS = 100

# --- SA Parameters ---
INITIAL_TEMPERATURE = 100.0
FINAL_TEMPERATURE = 0.01
COOLING_RATE = 0.995
STEPS_PER_TEMPERATURE = 100

# --- Test Data Parameters ---
NUM_TEST_PICKLISTS = 50
ITEMS_PER_PICKLIST = 10

# -----------------------------------------------------------------
#  1. NEW: CREATE MOCK SALES DATA
# -----------------------------------------------------------------
def create_sales_data(num_products):
    """Creates a DataFrame with product IDs and sales data (A, B, C categories)."""
    print("Creating mock sales data with ABC categories...")
    sales_data = np.random.exponential(scale=100, size=num_products).astype(int) + 10
    product_ids = [f"SKU_{i}" for i in range(num_products)]
    
    df = pd.DataFrame({'ProductID': product_ids, 'MonthlySales': sales_data})
    df = df.sort_values(by='MonthlySales', ascending=False).reset_index(drop=True)
    
    df['CumulativeShare'] = df['MonthlySales'].cumsum() / df['MonthlySales'].sum()
    
    def classify_product(row):
        if row['CumulativeShare'] <= 0.80: return 'A'
        elif row['CumulativeShare'] <= 0.95: return 'B'
        else: return 'C'
            
    df['Category'] = df.apply(classify_product, axis=1)
    
    # Save the sales data so our visualizer can use it later
    df.to_csv('product_sales_data.csv', index=False)
    print("Sales data saved to 'product_sales_data.csv'\n")
    return df

# -----------------------------------------------------------------
#  2. NEW: WEIGHTED PICK LIST GENERATOR
# -----------------------------------------------------------------
def generate_weighted_test_lists(product_df, num_lists, items_per_list):
    """Generates pick lists where A-items are more likely to be chosen."""
    print(f"Generating {num_lists} weighted sample pick lists...")
    product_ids = list(product_df['ProductID'])
    # Simple weighting: A-items are 10x more likely, B are 3x
    weights = product_df['Category'].map({'A': 10, 'B': 3, 'C': 1}).values.astype(float)
    weights /= weights.sum() # Normalize to probabilities
    
    test_pick_lists = []
    for _ in range(num_lists):
        # Use np.random.choice with p=weights
        chosen_products = np.random.choice(
            product_ids, 
            size=items_per_list, 
            replace=False, 
            p=weights
        )
        test_pick_lists.append(list(chosen_products))
    print("Weighted pick lists generated.\n")
    return test_pick_lists

# -----------------------------------------------------------------
#  3. TSP SOLVER (Unchanged)
# -----------------------------------------------------------------
def solve_tsp_nearest_neighbor(pick_list_coords):
    """Calculates the tour length for a single pick list using the Nearest Neighbor heuristic."""
    current_loc = pick_list_coords[0] # Start at the Dock
    unvisited = pick_list_coords[1:]
    total_distance = 0
    
    while unvisited:
        min_dist = float('inf')
        nearest_neighbor = None
        
        # Find the closest unvisited item
        for neighbor in unvisited:
            # Manhattan Distance
            dist = abs(current_loc[0] - neighbor[0]) + abs(current_loc[1] - neighbor[1])
            if dist < min_dist:
                min_dist = dist
                nearest_neighbor = neighbor
        
        total_distance += min_dist
        current_loc = nearest_neighbor
        unvisited.remove(nearest_neighbor)
        
    # Add final distance to return to the Dock
    total_distance += abs(current_loc[0] - 0) + abs(current_loc[1] - 0)
    return total_distance

# -----------------------------------------------------------------
#  4. FITNESS FUNCTION (Unchanged, but now gets better data)
# -----------------------------------------------------------------
def calculate_layout_fitness(layout, test_pick_lists):
    """Calculates the average TSP distance for a given layout over a sample of pick lists."""
    total_distance = 0
    dock_coord = (0, 0)
    
    for product_list in test_pick_lists:
        # Get the coordinates for the items in this list, based on the current layout
        pick_list_coords = [dock_coord]
        for product_id in product_list:
            if product_id in layout: # Check if product exists in layout
                pick_list_coords.append(layout[product_id])
            
        total_distance += solve_tsp_nearest_neighbor(pick_list_coords)
        
    # Return the average distance
    return total_distance / len(test_pick_lists)

# -----------------------------------------------------------------
#  5. HELPER FUNCTIONS (Unchanged)
# -----------------------------------------------------------------
def create_random_layout(product_ids, width, height):
    """Creates one random layout."""
    locations = [(y, x) for y in range(height) for x in range(width)]
    random.shuffle(locations)
    products_to_place = product_ids[:len(locations)]
    layout = {products_to_place[i]: locations[i] for i in range(len(products_to_place))}
    return layout

def create_neighbor_layout(layout):
    """Creates a "neighbor" layout by swapping two random products."""
    neighbor = copy.deepcopy(layout)
    p1, p2 = random.sample(list(neighbor.keys()), 2)
    loc1 = neighbor[p1]
    loc2 = neighbor[p2]
    neighbor[p1] = loc2
    neighbor[p2] = loc1
    return neighbor

# -----------------------------------------------------------------
#  6. SIMULATED ANNEALING ALGORITHM (Updated to use new functions)
# -----------------------------------------------------------------
def run_simulated_annealing():
    """Main SA optimization loop."""
    
    print("Initializing Simulated Annealing (V2 with Weighted Data)...")
    start_time = time.time()
    
    # --- Initialization ---
    # 1. Create sales data and get product lists
    product_df = create_sales_data(NUM_PRODUCTS)
    product_ids = list(product_df['ProductID'])
    
    # 2. Generate WEIGHTED test lists
    test_pick_lists = generate_weighted_test_lists(product_df, NUM_TEST_PICKLISTS, ITEMS_PER_PICKLIST)
    
    # 3. Create initial random layout
    current_layout = create_random_layout(product_ids, WAREHOUSE_WIDTH, WAREHOUSE_HEIGHT)
    current_cost = calculate_layout_fitness(current_layout, test_pick_lists)
    
    best_layout = copy.deepcopy(current_layout)
    best_cost = current_cost
    
    current_temp = INITIAL_TEMPERATURE
    iteration = 0
    
    print(f"Initial layout cost (from weighted lists): {best_cost:.2f}")
    print("Starting optimization loop...")

    # --- Main Loop ---
    while current_temp > FINAL_TEMPERATURE:
        for _ in range(STEPS_PER_TEMPERATURE):
            neighbor_layout = create_neighbor_layout(current_layout)
            neighbor_cost = calculate_layout_fitness(neighbor_layout, test_pick_lists)
            cost_difference = neighbor_cost - current_cost
            
            if cost_difference < 0:
                current_layout = neighbor_layout
                current_cost = neighbor_cost
            else:
                acceptance_prob = math.exp(-cost_difference / current_temp)
                if random.random() < acceptance_prob:
                    current_layout = neighbor_layout
                    current_cost = neighbor_cost
            
            if current_cost < best_cost:
                best_layout = copy.deepcopy(current_layout)
                best_cost = current_cost
        
        current_temp *= COOLING_RATE
        iteration += 1
        
        if iteration % 20 == 0:
            print(f"  Iter {iteration:3d} | Temp: {current_temp:6.2f} | Current Cost: {current_cost:8.2f} | Best Cost: {best_cost:8.2f}")

    # --- End of Loop ---
    end_time = time.time()
    print("\n...Optimization Finished.")
    print(f"Total time: {(end_time - start_time):.2f} seconds")
    print(f"Final Best Cost (Avg. Distance): {best_cost:.2f}")
    
    # 8. Save the final best layout
    np.save('layout_sa.npy', best_layout)
    print("\nFinal optimized layout saved to 'layout_sa.npy'")
    
    return best_layout

# --- Run the optimizer ---
if __name__ == "__main__":
    final_layout = run_simulated_annealing()
