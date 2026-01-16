import numpy as np
import json
import random
from typing import List, Dict

class ARCSanityGenerator:
    """Generates synthetic ARC-like tasks for sanity benchmarking."""
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

    def _gen_random_grid(self, rows, cols, max_colors=3):
        return np.random.randint(0, max_colors + 1, (rows, cols)).tolist()

    def generate_identity(self, num_samples=100, min_size=2, max_size=5):
        samples = []
        for _ in range(num_samples):
            r, c = random.randint(min_size, max_size), random.randint(min_size, max_size)
            grid = self._gen_random_grid(r, c)
            samples.append({
                "train": [{"input": grid, "output": grid}],
                "test": [{"input": grid, "output": grid}]
            })
        return samples

    def generate_color_permutation(self, num_samples=100, min_size=2, max_size=5):
        samples = []
        for _ in range(num_samples):
            r, c = random.randint(min_size, max_size), random.randint(min_size, max_size)
            grid = self._gen_random_grid(r, c)
            # Create a consistent mapping
            mapping = list(range(10))
            random.shuffle(mapping)
            
            def apply_map(g):
                return [[mapping[val] for val in row] for row in g]
                
            samples.append({
                "train": [{"input": grid, "output": apply_map(grid)}],
                "test": [{"input": grid, "output": apply_map(grid)}]
            })
        return samples

    def generate_translation(self, num_samples=100, size=7):
        samples = []
        for _ in range(num_samples):
            # Fixed 1x1 object translation for simplicity
            obj_color = random.randint(1, 9)
            input_grid = np.zeros((size, size), dtype=int)
            ir, ic = random.randint(0, size-2), random.randint(0, size-2)
            input_grid[ir, ic] = obj_color
            
            output_grid = np.zeros((size, size), dtype=int)
            output_grid[ir+1, ic+1] = obj_color
            
            samples.append({
                "train": [{"input": input_grid.tolist(), "output": output_grid.tolist()}],
                "test": [{"input": input_grid.tolist(), "output": output_grid.tolist()}]
            })
        return samples

    def generate_reflection(self, num_samples=100, size=5):
        samples = []
        for _ in range(num_samples):
            # 3x3 pattern in 5x5 grid
            pattern = self._gen_random_grid(3, 3)
            input_grid = np.zeros((size, size), dtype=int)
            input_grid[1:4, 1:4] = pattern
            
            # Vertical reflect
            output_grid = np.flip(input_grid, axis=0).tolist()
            
            samples.append({
                "train": [{"input": input_grid.tolist(), "output": output_grid}],
                "test": [{"input": input_grid.tolist(), "output": output_grid}]
            })
        return samples

    def generate_scaling(self, num_samples=100):
        samples = []
        for _ in range(num_samples):
            # 2x2 pattern scaled to 4x4
            pattern = np.random.randint(0, 4, (2, 2))
            input_grid = pattern.tolist()
            output_grid = np.repeat(np.repeat(pattern, 2, axis=0), 2, axis=1).tolist()
            
            samples.append({
                "train": [{"input": input_grid, "output": output_grid}],
                "test": [{"input": input_grid, "output": output_grid}]
            })
        return samples

    def save_to_jsonl(self, samples, path):
        with open(path, 'w') as f:
            for s in samples:
                f.write(json.dumps(s) + '\n')

if __name__ == "__main__":
    gen = ARCSanityGenerator()
    tasks = {
        "identity": gen.generate_identity(),
        "color": gen.generate_color_permutation(),
        "translation": gen.generate_translation()
    }
    import os
    os.makedirs("data/arc_sanity", exist_ok=True)
    for name, s in tasks.items():
        gen.save_to_jsonl(s, f"data/arc_sanity/{name}.jsonl")
