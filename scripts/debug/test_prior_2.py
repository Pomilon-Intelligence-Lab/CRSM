import torch
import sys
import os

sys.path.append(os.getcwd() + "/CRSM")
from crsm.core.reasoning import AsyncDeliberationLoop, ARCContextEvaluator

prompt_tokens = [16, 17, 46, 47, 1, 2, 13, 3, 4, 13, 14, 16, 17, 46, 47, 1, 2, 13, 3, 4, 13, 14, 12]
evaluator = ARCContextEvaluator(prompt_tokens)

loop = AsyncDeliberationLoop(None)
loop.path_evaluator = evaluator

logits = torch.zeros(1, 100)
loop._current_seq_list = prompt_tokens
print("Step 0 (len=0) max token:", torch.argmax(loop._apply_structural_prior(logits.clone(), prompt_tokens + [])[0]).item())

loop._current_seq_list = prompt_tokens + [97]
print("Step 1 (len=1) max token:", torch.argmax(loop._apply_structural_prior(logits.clone(), prompt_tokens + [97])[0]).item())

loop._current_seq_list = prompt_tokens + [97, 17]
print("Step 2 (len=2) max token:", torch.argmax(loop._apply_structural_prior(logits.clone(), prompt_tokens + [97, 17])[0]).item())
