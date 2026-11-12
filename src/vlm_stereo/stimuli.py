# Random Dot Stereogram (RDS) Generator
# Must account for VLM's encoder patch size 
# Qwen2.5-VL uses 14 x 14 pxl patches 
import numpy as np
from PIL import Image 

def make_rds(img_size = 448, sq_size = 224, dot_size = 4, density = 0.5, disparity = 0, seed = None):
    """Return (left, right) as uint8 arrays with values of 0 or 255"""

    rng = np.random.default_rng(seed)

    # Create a random dot pattern
    pattern = rng.choice([0, 255], size=(img_size, img_size), p=[1-density, density]).astype(np.uint8)

    # Plot initial image
    Image.fromarray(pattern).show()




