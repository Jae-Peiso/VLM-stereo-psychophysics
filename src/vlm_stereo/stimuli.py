# Random Dot Stereogram (RDS) Generator
# Must account for VLM's encoder patch size 
# Qwen2.5-VL uses 14 x 14 pxl patches 
import numpy as np

def make_rds(img_size = 448, sq_size = 224, dot_size = 4, density = 0.5, disparity = 0, seed = None):
    """Return (left, right) as uint8 arrays with values of 0 or 255"""

    rng = np.random.default_rng(seed)
    low, high = -dot_size + 1, img_size
    span = high - low
    n_dots = int(round(-span**2 * np.log(1-density) / dot_size**2))
    dot_positions = rng.integers(low, high, size=(n_dots, 2)) # columns: (row, col)

    # Add dots (stamping dots)
    img = np.full((img_size, img_size), 255, dtype=np.uint8) # white background image

    # Loop-to-stamp dots (imgA)
    imgA = img.copy()
    for row, col in dot_positions: 
        r0, c0 = max(row, 0), max(col, 0)
        r1, c1 = row + dot_size, col + dot_size
        imgA[r0:r1, c0:c1] = 0 # black dot

    return imgA

        
      # Vectorized stamping of dots (imgB)




   





