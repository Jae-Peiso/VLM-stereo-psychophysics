# Random Dot Stereogram (RDS) Generator
# Must account for VLM's encoder patch size 
 
import numpy as np

# Main Function: Random Dot Stereogram Pair Generator (not fully functional)
def make_rds(height = 448, width = 448, sq_size = 224, dot_size = 4, density = 0.5, disparity = 0, seed = None):
    rng = np.random.default_rng(seed)
    shape = (height, width)
    positions = _random_positions(rng, shape, density, dot_size)
    img = _stamp_vectorized(positions, dot_size, shape)

    return img


# Helper Function: Generating Random Positions 
def _random_positions(rng, shape, density, dot_size):
    """Return (left, right) as uint8 arrays with values of 0 or 255"""

    low = -dot_size + 1
    span_r, span_c = shape[0] - low, shape[1] - low 
    n_dots = int(round(-span_r*span_c * np.log(1-density) / dot_size**2))
    rows = rng.integers(low, shape[0], size=(n_dots)) # rows
    cols = rng.integers(low, shape[1], size=(n_dots)) # columns

    return np.stack((rows, cols), axis=1)


# Non-vectorized Dot Stamper (Not used by make_rds()--For testing purposes only) 
def _stamp_loop(positions, dot_size, shape):

    # Add dots (stamping dots)
    img = np.full(shape, 255, dtype=np.uint8) # white background image

    # Loop-to-stamp dots (img)
    for row, col in positions: 
        r0, c0 = max(row, 0), max(col, 0)
        r1, c1 = row + dot_size, col + dot_size
        img[r0:r1, c0:c1] = 0 # black dot

    return img

    
# Vectorized Dot Stamper (Used by make_rds())
def _stamp_vectorized(positions, dot_size, shape):

    img = np.full(shape, 255, dtype=np.uint8) # white background image

    rows0, cols0 = positions[:, 0], positions[:, 1] # all dots' corners
    bounds = np.array(shape)[:, None] 

    for dr in range(dot_size):
        for dc in range(dot_size):
            rows = rows0 + dr # this pixel slot, for every dot at once
            cols = cols0 + dc
            pts = np.array([rows, cols]) # shape (2, n)
            inside = ((pts >= 0) & (pts < bounds)).all(axis=0) #shape (n,) 
            img[rows[inside], cols[inside]] = 0 # black dot

    return img



   





