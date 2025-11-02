# visualize_layout.py
# Loads a saved layout and plots it as a heatmap.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def visualize_saved_layout(layout_file, sales_data_file, title):
    """Loads and plots the warehouse layout."""
    
    # 1. Load the sales data
    try:
        sales_df = pd.read_csv(sales_data_file)
        # Create a dictionary for easy lookup: {'SKU_0': 'A', 'SKU_1': 'C', ...}
        product_to_category = pd.Series(sales_df.Category.values, index=sales_df.ProductID).to_dict()
    except FileNotFoundError:
        print(f"Error: Could not find {sales_data_file}.")
        print("Please run the optimizer script (sa_optimizer_v2.py) first to generate this file.")
        return

    # 2. Load the saved layout
    try:
        # layout is a dictionary {'SKU_0': (y, x), ...}
        layout = np.load(layout_file, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Error: Could not find {layout_file}.")
        print("Please run the optimizer script first to generate this file.")
        return

    # 3. Create the heatmap grid
    # Get warehouse dimensions from the layout coordinates
    max_y = max(loc[0] for loc in layout.values())
    max_x = max(loc[1] for loc in layout.values())
    width = max_x + 1
    height = max_y + 1
    
    # Create an empty grid (filled with 0s)
    grid = np.zeros((height, width))
    
    # Create a mapping for heatmap colors (A=3, B=2, C=1)
    category_to_value = {'A': 3, 'B': 2, 'C': 1}
    
    # 4. Populate the grid
    for product_id, location in layout.items():
        y, x = location
        category = product_to_category.get(product_id, 'C') # Default to 'C' if missing
        grid[y, x] = category_to_value.get(category, 1)

    # 5. Plot the heatmap
    print(f"Plotting heatmap for {title}...")
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Use 'hot' colormap: yellow=hot(A), red=warm(B), black=cold(C)
    im = ax.imshow(grid, cmap='hot', interpolation='nearest')
    
    # Create a colorbar
    cbar = ax.figure.colorbar(im, ax=ax, ticks=[1, 2, 3])
    cbar.ax.set_yticklabels(['C-Items (Cold)', 'B-Items (Warm)', 'A-Items (Hot)'])
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Aisle")
    ax.set_ylabel("Shelf")
    
    # Display the grid lines
    ax.set_xticks(np.arange(-.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-.5, height, 1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    
    plt.show()

# --- Run the visualizer ---
if __name__ == "__main__":
    visualize_saved_layout(
        layout_file='layout_sa.npy',
        sales_data_file='product_sales_data.csv',
        title='Optimized Layout (Simulated Annealing)'
    )
