# app.py: Warehouse Layout Optimization Project

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_mock_data(num_products=100):
    """
    Creates a mock dataset of products with sales volumes that follow a Pareto-like distribution.
    This simulates a real-world scenario where a few products are best-sellers.
    """
    print("Step 1: Creating mock product sales data...")
    # Create an exponential distribution for sales - this ensures some are high, many are low
    sales_data = np.random.exponential(scale=100, size=num_products).astype(int) + 10
    product_ids = [f"SKU{1000+i}" for i in range(num_products)]
    
    df = pd.DataFrame({
        'ProductID': product_ids,
        'MonthlySales': sales_data
    })
    
    # Sort by sales to make it easier to see the distribution
    df = df.sort_values(by='MonthlySales', ascending=False).reset_index(drop=True)
    print("Mock data created successfully.\n")
    return df

def perform_abc_analysis(df):
    """
    Performs ABC analysis on the product data.
    - A items: Top 80% of sales
    - B items: Next 15% of sales
    - C items: Last 5% of sales
    """
    print("Step 2: Performing ABC Analysis...")
    df['SalesShare'] = df['MonthlySales'] / df['MonthlySales'].sum()
    df['CumulativeShare'] = df['SalesShare'].cumsum()
    
    def classify_product(row):
        if row['CumulativeShare'] <= 0.80:
            return 'A'
        elif row['CumulativeShare'] <= 0.95:
            return 'B'
        else:
            return 'C'
            
    df['Category'] = df.apply(classify_product, axis=1)
    print("ABC Analysis complete. Product categories assigned.\n")
    return df

def generate_naive_layout(products_list, width, height):
    """
    Generates a naive (random) warehouse layout.
    """
    print("Step 3: Generating Naive (Random) Layout...")
    if len(products_list) > width * height:
        raise ValueError("Not enough space in the warehouse for all products.")
        
    layout = np.zeros((height, width))
    product_map = {'A': 3, 'B': 2, 'C': 1} # Numerical map for heatmap colors
    
    # Shuffle the products randomly
    np.random.shuffle(products_list)
    
    # Place products in the grid
    for i in range(height):
        for j in range(width):
            if i * width + j < len(products_list):
                product_category = products_list[i * width + j]
                layout[i, j] = product_map[product_category]
    print("Naive layout generated.\n")
    return layout

def generate_optimized_layout(a_items, b_items, c_items, width, height):
    """
    Generates an optimized warehouse layout using the Zoning Heuristic.
    """
    print("Step 4: Generating Optimized Layout using Zoning Heuristic...")
    layout = np.zeros((height, width))
    product_map = {'A': 3, 'B': 2, 'C': 1}
    
    # Create a list of all coordinates and sort them by distance from the shipping dock (0,0)
    # Using Manhattan distance: |x1-x2| + |y1-y2|
    coords = [(y, x) for y in range(height) for x in range(width)]
    coords.sort(key=lambda c: c[0] + c[1])
    
    # Place items based on zones
    current_coord_index = 0
    
    # Place A-items in the closest (Gold) zone
    for item in a_items:
        y, x = coords[current_coord_index]
        layout[y, x] = product_map[item]
        current_coord_index += 1
        
    # Place B-items in the next (Silver) zone
    for item in b_items:
        y, x = coords[current_coord_index]
        layout[y, x] = product_map[item]
        current_coord_index += 1
        
    # Place C-items in the furthest (Bronze) zone
    for item in c_items:
        y, x = coords[current_coord_index]
        layout[y, x] = product_map[item]
        current_coord_index += 1
        
    print("Optimized layout generated.\n")
    return layout

def visualize_layout(layout_grid, title):
    """
    Creates and displays a heatmap of the given warehouse layout grid.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    # Use a color map where hot colors (red/yellow) are high values (A-items)
    im = ax.imshow(layout_grid, cmap='hot', interpolation='nearest')
    
    # Create a colorbar
    cbar = ax.figure.colorbar(im, ax=ax, ticks=[1, 2, 3])
    cbar.ax.set_yticklabels(['C-Items (Cold)', 'B-Items (Warm)', 'A-Items (Hot)'])
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Aisle")
    ax.set_ylabel("Shelf")
    
    # Display the grid lines
    ax.set_xticks(np.arange(-.5, layout_grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, layout_grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)
    
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # --- Parameters You Can Change ---
    WAREHOUSE_WIDTH = 10
    WAREHOUSE_HEIGHT = 10
    NUM_PRODUCTS = 100
    
    # 1. Create Data
    product_df = create_mock_data(num_products=NUM_PRODUCTS)
    
    # 2. Perform Analysis
    product_df = perform_abc_analysis(product_df)
    
    # 3. Separate item lists for the layout functions
    all_products_list = list(product_df['Category'])
    a_items_list = list(product_df[product_df['Category'] == 'A']['Category'])
    b_items_list = list(product_df[product_df['Category'] == 'B']['Category'])
    c_items_list = list(product_df[product_df['Category'] == 'C']['Category'])
    
    # 4. Generate the two layouts
    naive_grid = generate_naive_layout(all_products_list, WAREHOUSE_WIDTH, WAREHOUSE_HEIGHT)
    optimized_grid = generate_optimized_layout(a_items_list, b_items_list, c_items_list, WAREHOUSE_WIDTH, WAREHOUSE_HEIGHT)
    
    # 5. Visualize both layouts
    print("Step 5: Visualizing layouts... (Two plot windows will open)")
    visualize_layout(naive_grid, "Naive (Random) Warehouse Layout")
    visualize_layout(optimized_grid, "Optimized Warehouse Layout (Zoning Heuristic)")
