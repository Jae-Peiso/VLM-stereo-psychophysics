import numpy as np
import pytest

from vlm_stereo.stimuli import (make_rds, _random_positions, _stamp_loop, _stamp_vectorized)

TOP = LEFT = 112 #square corner for defualts (448 image, 224 square)
SQ = 224

# Testing for correct output format
def test_output_format():
    L, R = make_rds(seed = 0, disparity = 8)
    for img in (L, R):
        assert img.shape == (448, 448)                                  # checking the shape
        assert img.dtype == np.uint8                                    # checking the data type
        assert set(np.unique(img)) <= {0, 255}                          # checking pixel values are black and white

# Testing that the same seed reproduces the same output and that a different seed does not reproduce the same results
def test_same_seed_same_and_diff_seed_diff():
    a = make_rds(seed = 1, disparity = 8)
    b = make_rds(seed = 1, disparity = 8)
    c = make_rds(seed = 2, disparity = 8)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])    # a & b are the same
    assert not np.array_equal(a[0], c[0])                               # a & c are NOT the same

# Testing that 0 disparity yields two identical images
def test_zero_disparity_gives_identical_images():
    L, R = make_rds(seed = 0, disparity = 0)
    assert np.array_equal(L, R)

# Testing disparity sign and magnitude
@pytest.mark.parametrize("d", [2, 4, 8, 16, -2, -4, -8, -16])
@pytest.mark.parametrize("dot_size", [1,4])
def test_disparity_sign_and_magnitude(d, dot_size):
    """Inside the square, left[x] == right[x - d]."""
    L, R = make_rds(seed = 0, disparity = d, dot_size = dot_size)
    mr = dot_size                                                       # rows are not impacted by disparity shift
    mc = dot_size + abs(d)                                              # stay clear of ragged edges and strips for horizontal shift
    rows = slice(TOP + mr, TOP + SQ - mr)                               # This is taking from inside of the square not the edges
    a = L[rows, LEFT + mc : LEFT + SQ - mc] 
    b = R[rows, LEFT + mc - d : LEFT + SQ - mc - d]                     # Shifting the slice by the disparity to ensure that a == b
    assert np.array_equal(a, b)

# Testing that strip refills are NOT the same.  
def test_refill_strip_independence():
    """Left eye's strip (left of square) vs. right eye's strip (right of square).
    Reusing the same random draw for both would make these identical."""
    s = 8 # each eye's shift
    L, R = make_rds(seed=0, disparity = 2 * s)
    strip_L = L[TOP:TOP + SQ, LEFT:LEFT + s] 
    strip_R = R[TOP:TOP + SQ, LEFT + SQ - s:LEFT + SQ]
    assert not np.array_equal(strip_L, strip_R)

# Testing loop stamper vs vectorized stamper, they should be the same. 
@pytest.mark.parametrize("shape, dot", [
    ((448, 420), 4),                                                    # non-square: catches row/column swaps 
    ((100, 100), 3),                                                    # odd dot size
    ((64, 80), 1)                                                       # 1 px dots
])

def test_stamping_versions_agree(shape, dot):
    rng = np.random.default_rng(0)
    pos = _random_positions(rng, shape, 0.5, dot)
    assert np.array_equal(_stamp_loop(pos, dot, shape),
                          _stamp_vectorized(pos, dot, shape))

# Testing that the actual coverage matches requested density (default is 0.5). Checking at disparity = to 0 and 16
@pytest.mark.parametrize("d", [0, 16])
def test_coverage_matches_density(d):
    cov = [(img == 0).mean()                                           # checking how much of the image is black
        for s in range(100)
        for img in make_rds(seed = s, disparity = d)]
    assert abs(np.mean(cov) - 0.5) < 0.001                             # ~6 SE, from measured STD of ~0.0016

# Testing that the images's edge density is the same as the images's interior (default is 0.5). 
@pytest.mark.parametrize("dot_size", [4, 8])
def test_edges_match_interior(dot_size):
    imgs = [make_rds(seed = s, dot_size = dot_size)[0] == 0 for s in range(50)]
    cols = np.mean([img.mean(axis = 0) for img in imgs], axis = 0)
    rows = np.mean([img.mean(axis = 1) for img in imgs], axis = 0)
    for profile in (cols, rows): 
        assert abs(profile[:4].mean() - 0.5) < 0.02
        assert abs(profile[-4:].mean() - 0.5) < 0.02

# Testing that the refilled strip has the same density as the rest of the image. 
def test_no_density_seam_at_strip():
    s = 8                                                                   # Single eye shift (disparity/2)
    strip = np.mean([(make_rds(seed = k, disparity = 2 * s)[0]
                 [TOP : TOP + SQ, LEFT : LEFT + s] == 0).mean()         # Slicing out just the strip
                 for k in range(100)])
    assert abs(strip - 0.5) < 0.02

# Testing that bad values raise ValueError 
@pytest.mark.parametrize("kwargs", [
    {"disparity":   3},                                                 # Odd disparity 
    {"disparity":   2.0},                                               # Non-integer
    {"disparity":   226},                                               # square + shift is too large
    {"sq_size":     448},                                               # square as large as image
    {"sq_size":     0},                                                 # area = 0 (no square)
    {"dot_size":    0},                                                 # area = 0 (no dots)
    {"dot_size":    2.5},                                               # Non-integer
    {"density":     0},                                                 # density = 0 (no dots)                        
    {"density":     1},                                                 # density = 1 (black out - no dots)
])
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        make_rds(**kwargs)