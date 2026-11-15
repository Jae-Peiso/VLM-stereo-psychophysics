# Random Dot Stereogram (RDS) Generator
# Must account for VLM's encoder patch size 
 
import numpy as np

# Main Function: Random Dot Stereogram Pair Generator (not fully functional)
def make_rds(height = 448, width = 448, sq_size = 224, dot_size = 4, density = 0.5, disparity = 0, seed = None):
    # Input Validation
    for name, value in [("height", height), ("width", width),
                        ("sq_size", sq_size), ("dot_size", dot_size),
                        ("disparity", disparity)]:
        _require_int(name, value) 

    # height must be greater than 0
    if height < 0:
        raise ValueError(f"height must be an integer larger than 0, got {height}.")

    # width must be greater than 0
    if width < 0:
        raise ValueError(f"width must be an integer larger than 0, got {width}.")
   
    # disparity must be an even integer.
    if disparity % 2 != 0:
        raise ValueError(f"Disparity must be an even integer, got {disparity}.")
   
    # sq_size must be larger than 0 and smaller than the smallest dimension (height or width)
    if not 0 < sq_size < min(height, width):
        raise ValueError(f"Square size {sq_size} must be between 0 and {min(height, width)} (exclusive).")

  
    # half is the disparity within an eye or each eyes' shift
    half = abs(disparity) // 2
    # (top, left) is the coordinate of the square's top left corner. Dividing by 2, halves the margins ensuring that the square is centered. 
    top = (height - sq_size) // 2
    left = (width - sq_size) // 2
    # Shape of the stimulus 
    shape = (height, width) 

    # sq_size + disparity must fit within height and width
    if left - half < 0 or left + sq_size + half > width: 
        raise ValueError(f"Square ({sq_size}px) shifted by {half}px does not fit in the width {width}px.")
    
    # dot_size has to be an integer from 1 to 1 px smaller than sq_size
    if not 1 <= dot_size < sq_size:
        raise ValueError(f"dot_size must be an integer from 1 to {sq_size - 1} (smaller than the square), got {dot_size}.")
    
    # density has to be a float between 0 and 1, exclusive. 
    if not 0 < density < 1:
        raise ValueError(f"density must be between 0 and 1 (exclusive), got {density}.")

    # getting random positions and stamping dots 
    rng = np.random.default_rng(seed)
    positions = _random_positions(rng, shape, density, dot_size)
    in_sq = _in_region(positions, dot_size, top, left, sq_size, sq_size)
    sq_dots = positions[in_sq]
    bg_dots = positions[~in_sq]

    shift_L = disparity // 2
    shift_R = -disparity // 2
    posL = _eye_positions(rng, bg_dots, sq_dots, shift_L, top, left, sq_size, dot_size, density) 
    posR = _eye_positions(rng, bg_dots, sq_dots, shift_R, top, left, sq_size, dot_size, density) 

    imgL = _stamp_vectorized(posL, dot_size, shape)
    imgR = _stamp_vectorized(posR, dot_size, shape)

   
    return imgL, imgR

# Helper Function: Check that inputs are integers 
def _require_int(name, value):
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}.") 
    # {!r} shows the user the values's representation, e.g., printing strings as '2' with quotes. 
 
# Helper Function: Generating Random Positions 
def _random_positions(rng, shape, density, dot_size):
    
    low = -dot_size + 1
    span_r, span_c = shape[0] - low, shape[1] - low 
    n_dots = int(round(-span_r*span_c * np.log(1-density) / dot_size**2))
    rows = rng.integers(low, shape[0], size=(n_dots)) # rows
    cols = rng.integers(low, shape[1], size=(n_dots)) # columns

    return np.stack((rows, cols), axis=1)

# Helper Function: Define what dots occur within the square
# Edge coordinates: pixel i occupies the interval [i, i + 1); a dot starting at corner r spans [r, r + dot_size).
def _in_region(positions, dot_size, top, left, region_h, region_w):
    centers = positions + dot_size / 2
   
    return ((centers[:,0] >= top) & (centers[:,0] < top + region_h) & 
            (centers[:,1] >= left) & (centers[:,1] < left + region_w))
    
# Helper Function: Shift, Occlude, Drop, & Refill 
def _eye_positions(rng, bg_dots, sq_dots, shift, top, left, sq_size, dot_size, density):
    """Dot positions for one eye: background minus the occluded part, plus the square shifted by 'shift' columns (refill added later)."""
    # 1. Move sq_dots horizontally by shift
    sq_shifted = sq_dots + np.array([0, shift])

    # 2. Find bg_dots now covered by the moved square and drop them. 
    covered = _in_region(bg_dots, dot_size, top, left + shift, sq_size, sq_size)
    bg_visible = bg_dots[~covered] 

    # 3. (Later: refill the strip the square shifted away from.)

    # 4. This eye's full set of dots. 
    return np.vstack([bg_visible, sq_shifted])

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



   





