import torch
import sys
import os

sys.path.append(os.getcwd() + "/CRSM")
from crsm.core.reasoning import AsyncDeliberationLoop, ARCContextEvaluator

# Mocking the context evaluator
# Let's say we have [2x2] -> [2x2], [2x2] -> [2x2] grids.
# We just need prompt_tokens to contain at least two pairs.
prompt_tokens = [
    # Train input 1
    16, 17, 46, 47, 1, 2, 13, 3, 4, 13, 14,
    # Train output 1
    16, 17, 46, 47, 1, 2, 13, 3, 4, 13, 14,
    # Train input 2
    16, 17, 46, 47, 1, 2, 13, 3, 4, 13, 14,
    # Train output 2
    16, 17, 46, 47, 1, 2, 13, 3, 4, 13, 14,
    # Test input
    16, 17, 46, 47, 1, 2, 13, 3, 4, 13, 14,
    # Output start
    12
]

evaluator = ARCContextEvaluator(prompt_tokens)
print("Dim changes:", evaluator.dim_changes)

loop = AsyncDeliberationLoop(None)
loop.path_evaluator = evaluator

logits = torch.zeros(1, 100)

# Step 0: path = []
loop._current_seq_list = prompt_tokens
biased = loop._apply_structural_prior(logits.clone(), prompt_tokens + [])
print("Step 0 max token:", torch.argmax(biased[0]).item())

# Step 1: path = [97] (suppose 97 was chosen)
biased = loop._apply_structural_prior(logits.clone(), prompt_tokens + [97])
print("Step 1 (after 97) max token:", torch.argmax(biased[0]).item())

# Step 2: path = [97, 17]
biased = loop._apply_structural_prior(logits.clone(), prompt_tokens + [97, 17])
print("Step 2 (after 97, 17) max token:", torch.argmax(biased[0]).item())

# Step 3: path = [17, 47] (assuming normal flow)
biased = loop._apply_structural_prior(logits.clone(), prompt_tokens + [17, 47])
print("Normal Step 2 (after 17, 47) max token:", torch.argmax(biased[0]).item())

