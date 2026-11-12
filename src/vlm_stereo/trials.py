import numpy as np

def pattern_seeds(master_seed, n_patterns):
    children = np.random.SeedSequence(master_seed).spawn(n_patterns)
    return [int(c.generate_state(1)[0]) for c in children]