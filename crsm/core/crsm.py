"""
Main CRSM (Continuous Reasoning State Model) implementation.
This module integrates the Mamba SSM backbone with the asynchronous MCTS deliberation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import asyncio
import os
import json
import uuid
import math
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass
from ..training.logger import logger

from .mamba import MambaModel
from .reasoning import AsyncDeliberationLoop
from .dynamics import LatentDynamics

@dataclass
class CRSMConfig:
    """Configuration class for CRSM model."""
    vocab_size: int
    hidden_size: int = 256  # d_model
    intermediate_size: int = 1024  # d_ffn
    num_hidden_layers: int = 4
    num_attention_heads: int = 4  # Not directly used but kept for compatibility
    max_position_embeddings: int = 2048
    d_state: int = 64
    dropout: float = 0.1
    c_puct: float = 1.0
    n_simulations: int = 50
    autonomous_mode: bool = False
    temperature: float = 0.8  # NEW: Sampling temperature
    top_k: int = 50  # NEW: Top-k sampling
    top_p: float = 0.95  # NEW: Nucleus sampling
    delta_decay: float = 0.9  # NEW: Decay factor for lagged deltas
    max_lag: int = 10  # NEW: Maximum lag to accept deltas
    delta_scale: float = 0.1  # Deprecated in favor of injection_rate, kept for compat
    injection_rate: float = 0.05  # NEW: Gated injection rate (alpha)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'CRSMConfig':
        return cls(**config_dict)
    
    def to_dict(self) -> Dict:
        return {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'intermediate_size': self.intermediate_size,
            'num_hidden_layers': self.num_hidden_layers,
            'num_attention_heads': self.num_attention_heads,
            'max_position_embeddings': self.max_position_embeddings,
            'd_state': self.d_state,
            'dropout': self.dropout,
            'c_puct': self.c_puct,
            'n_simulations': self.n_simulations,
            'autonomous_mode': self.autonomous_mode,
            'temperature': self.temperature,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'delta_decay': self.delta_decay,
            'max_lag': self.max_lag,
            'delta_scale': self.delta_scale,
            'injection_rate': self.injection_rate,
        }

class CRSMModel(nn.Module):
    """CRSM model with HuggingFace-style interface"""
    def __init__(self, config: CRSMConfig):
        super().__init__()
        self.config = config
        self.crsm = CRSM(
            vocab_size=config.vocab_size,
            d_model=config.hidden_size,
            d_state=config.d_state,
            d_ffn=config.intermediate_size,
            num_layers=config.num_hidden_layers,
            dropout=config.dropout,
            c_puct=config.c_puct,
            n_simulations=config.n_simulations,
            autonomous_mode=config.autonomous_mode,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            delta_decay=config.delta_decay,
            max_lag=config.max_lag,
            delta_scale=getattr(config, 'delta_scale', 0.1),
            injection_rate=getattr(config, 'injection_rate', 0.05),
        )

    def load_dynamics(self, dynamics_path: str):
        """Load trained dynamics model."""
        self.crsm.load_dynamics(dynamics_path)
        
    def forward(self, input_ids, attention_mask=None, states=None):
        # Forward pass with optional states
        logits, new_states = self.crsm(input_ids, states)
        return logits, new_states
        
    @classmethod
    def from_pretrained(cls, path: str) -> 'CRSMModel':
        """Load from pretrained checkpoint"""
        with open(f"{path}/config.json", 'r') as f:
            config_dict = json.load(f)
        config = CRSMConfig.from_dict(config_dict)
        model = cls(config)
        model.load_state_dict(torch.load(f"{path}/pytorch_model.bin"))
        return model
    
    def save_pretrained(self, path: str):
        """Save model and config"""
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/config.json", 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        torch.save(self.state_dict(), f"{path}/pytorch_model.bin")

class CRSM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 256,
                 d_state: int = 64,
                 d_ffn: int = 1024,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 c_puct: float = 1.0,
                 n_simulations: int = 50,
                 autonomous_mode: bool = False,
                 temperature: float = 0.8,
                 top_k: int = 50,
                 top_p: float = 0.95,
                 delta_decay: float = 0.9,
                 max_lag: int = 10,
                 delta_scale: float = 0.1,
                 injection_rate: float = 0.05):
        """
        Args:
            vocab_size: Size of the vocabulary
            d_model: Model dimension
            d_state: State dimension for SSM
            d_ffn: Feed-forward network dimension
            num_layers: Number of Mamba blocks
            dropout: Dropout rate
            c_puct: Exploration constant for MCTS
            n_simulations: Number of MCTS simulations per deliberation
            autonomous_mode: Whether to enable autonomous operation
            temperature: Sampling temperature for generation
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            delta_decay: Decay factor for lagged deltas
            max_lag: Maximum lag to accept deltas
            delta_scale: Scaling factor (Deprecated)
            injection_rate: Gated injection rate (alpha)
        """
        super().__init__()
        
        # Mamba SSM backbone
        self.backbone = MambaModel(
            vocab_size=vocab_size,
            d_model=d_model,
            d_state=d_state,
            d_ffn=d_ffn,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Dynamics model
        self.dynamics = LatentDynamics(d_model=d_model, num_layers=num_layers)
        
        # Reasoning module
        self.reasoning = AsyncDeliberationLoop(
            mamba_model=self.backbone,
            c_puct=c_puct,
            n_simulations=n_simulations
        )
        
        # Connect dynamics to reasoning
        self.reasoning.dynamics_model = self.dynamics
        self.reasoning.temperature = temperature
        self.reasoning.use_sampling = True
        
        # Sampling parameters
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.delta_decay = delta_decay
        self.max_lag = max_lag
        self.delta_scale = delta_scale
        self.injection_rate = injection_rate
        
        self.autonomous_mode = autonomous_mode
        self._thinking_task = None
        # Background suggestion infrastructure
        self._bg_task = None
        # create queue and lock eagerly to avoid races
        self._bg_queue = asyncio.Queue(maxsize=8)

        # Canonical latent state and lock for concurrency
        self.latent_state = None
        self._state_lock = asyncio.Lock()
        
        # Background deliberation components (initialized lazily)
        self._deliberation_task = None
        self._deliberation_requests = None
        self._deliberation_results = {}
        self._stop_deliberation = False
        self.state_update_queue = asyncio.Queue()
        self.current_generation_id = None
        self._current_step_index = 0  # Track step for lag calculation
        
        # Targeted Delta Buffer: Ensures deltas are applied at the exact step they were planned for.
        self._targeted_deltas = {}
    
    def load_dynamics(self, dynamics_path: str):
        """Load trained dynamics model."""
        device = next(self.parameters()).device
        ckpt = torch.load(dynamics_path, map_location=device)
        
        if 'dynamics_state' in ckpt:
            self.dynamics.load_state_dict(ckpt['dynamics_state'])
            logger.info(f"✓ Loaded dynamics from {dynamics_path}")
        elif isinstance(ckpt, dict) and all(k.startswith('net.') for k in ckpt.keys()):
            self.dynamics.load_state_dict(ckpt)
            logger.info(f"✓ Loaded dynamics from {dynamics_path}")
        else:
            logger.warning(f"⚠ Warning: Could not load dynamics from {dynamics_path}")
            return False
        
        # Update reasoning module reference
        self.reasoning.dynamics_model = self.dynamics
        logger.info("✓ Connected dynamics to reasoning module")
        return True
        
    def forward(self, 
                x: torch.Tensor, 
                states: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the model
        Args:
            x: Input tensor of token indices
            states: Optional list of previous states
        Returns:
            logits: Output logits
            new_states: Updated states
        """
        return self.backbone(x, states)

    def init_latent_state(self, batch_size: int = 1, device: Optional[torch.device] = None):
        """Initialize the canonical latent state for the model."""
        device = device or next(self.parameters()).device
        self.latent_state = self.backbone.init_state(batch_size=batch_size, device=device)
        return self.latent_state

    def apply_state_delta(self, delta, scale: Optional[Union[float, List[float]]] = None):
        """
        Apply state update from deliberation.
        
        NOTE: 'delta' here is now interpreted as the 'TARGET STATE' from MCTS, 
        not a perturbation vector, when using Gated Injection.
        
        Formula: state = (1 - alpha) * state + alpha * target_state
        """
        if delta is None or self.latent_state is None:
            return
        
        # Default alpha
        base_alpha = self.injection_rate
        
        try:
            if isinstance(self.latent_state, list) and isinstance(delta, list):
                for i in range(min(len(self.latent_state), len(delta))):
                    if self.latent_state[i] is None or delta[i] is None:
                        continue
                        
                    # Determine alpha for this layer
                    if isinstance(scale, list) and i < len(scale):
                        alpha = scale[i]
                    elif isinstance(scale, (float, int)):
                        alpha = scale
                    else:
                        alpha = base_alpha
                        
                    # Gated Injection: (1 - alpha) * state + alpha * target
                    self.latent_state[i] = (1 - alpha) * self.latent_state[i] + alpha * delta[i]
            else:
                # Gated Injection for single tensor
                alpha = scale if isinstance(scale, (float, int)) else base_alpha
                self.latent_state = (1 - alpha) * self.latent_state + alpha * delta
        except Exception:
            return
    
    def sample_next_token(self, logits: torch.Tensor) -> int:
        """Sample next token with temperature and nucleus sampling."""
        logits = logits / self.temperature
        
        # Top-k filtering
        if self.top_k > 0:
            indices_to_remove = logits < torch.topk(logits, min(self.top_k, logits.size(-1)))[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) filtering
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()
    
    def _start_background_deliberation(self):
        """Start async MCTS worker."""
        if self._deliberation_task is not None:
            return  # Already running
        
        # Initialize components on first use
        if self._state_lock is None:
            self._state_lock = asyncio.Lock()
        
        if self.state_update_queue is None:
            self.state_update_queue = asyncio.Queue()
        
        if self._deliberation_requests is None:
            self._deliberation_requests = asyncio.Queue()
        
        self._deliberation_results = {}
        self._stop_deliberation = False
        
        self._deliberation_task = asyncio.create_task(self._deliberation_worker())
        logger.info("✓ Started background deliberation worker")

    def _ensure_async_components(self):
        """Ensure async components are initialized."""
        if not hasattr(self, '_deliberation_task'):
            self._deliberation_task = None
        if not hasattr(self, '_deliberation_requests'):
            self._deliberation_requests = None
        if not hasattr(self, '_deliberation_results'):
            self._deliberation_results = {}
        if not hasattr(self, '_state_lock'):
            self._state_lock = None
        if not hasattr(self, 'state_update_queue'):
            self.state_update_queue = None
        if not hasattr(self, '_stop_deliberation'):
            self._stop_deliberation = False

    async def _deliberation_worker(self):
        """Background MCTS loop - runs continuously."""
        logger.info("[Deliberation Worker] Started")
        
        while not self._stop_deliberation:
            try:
                # Get next deliberation request (non-blocking)
                request_item = await asyncio.wait_for(
                    self._deliberation_requests.get(), 
                    timeout=0.1
                )
                
                # Handle both old format (pos, seq) and new format (pos, seq, gen_id)
                gen_id = None
                if len(request_item) == 3:
                    position, sequence, gen_id = request_item
                else:
                    position, sequence = request_item
                
                # Copy current state (locked)
                async with self._state_lock:
                    state_copy = [s.clone() if s is not None else None 
                                for s in self.latent_state]
                    captured_step = self._current_step_index
                
                # =========================================================
                # FORWARD-PROJECTED PLANNING
                # =========================================================
                # If we are planning for a future position, we must project
                # the current state forward to that position.
                lag = position - captured_step
                if lag > 0:
                    # logger.debug(f"  [Deliberation] Projecting state forward {lag} steps...")
                    state_to_plan = await asyncio.to_thread(
                        self.reasoning.project_future_state,
                        state_copy,
                        lag
                    )
                else:
                    state_to_plan = state_copy

                # Run MCTS (this is the slow part, but it's in background)
                logger.debug(f"  [Deliberation] Planning for position {position}...")
                
                # Run on GPU but in separate async context
                action, delta, confidence = await asyncio.to_thread(
                    self.reasoning.deliberate_sync,
                    sequence,
                    state_to_plan
                )
                
                # Store result
                self._deliberation_results[position] = (action, confidence)
                
                # Queue state update (Now including confidence and captured_step)
                if delta is not None:
                    await self.state_update_queue.put((position, delta, gen_id, confidence, captured_step))
                
                # print(f"  [Deliberation] Completed for position {position}: token={action} (conf={confidence:.2f})")
                
            except asyncio.TimeoutError:
                # No requests, keep looping
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"[Deliberation Worker] Error: {e}")
                await asyncio.sleep(0.1)
        
        logger.info("[Deliberation Worker] Stopped")

    def _stop_background_deliberation(self):
        """Stop background worker."""
        self._stop_deliberation = True
        if self._deliberation_task is not None:
            self._deliberation_task.cancel()
            self._deliberation_task = None
        logger.info("✓ Stopped background deliberation worker")

    def _request_deliberation(self, position, sequence):
        """Request MCTS to deliberate on future token (non-blocking)."""
        try:
            self._deliberation_requests.put_nowait((position, sequence, self.current_generation_id))
        except asyncio.QueueFull:
            # Queue full, skip this request
            pass

    def _get_suggestion(self, position, timeout=0):
        """Get deliberation result if ready (non-blocking)."""
        if position in self._deliberation_results:
            result = self._deliberation_results.pop(position)
            return result
        return None

    async def _apply_pending_deltas(self, current_step=None):
        """Apply state corrections from deliberation (non-blocking)."""
        try:
            while not self.state_update_queue.empty():
                item = self.state_update_queue.get_nowait()

                # Handle format with/without gen_id and confidence
                # Queue items can be:
                # (pos, delta)
                # (pos, delta, gen_id)
                # (pos, delta, gen_id, confidence)
                # (pos, delta, gen_id, confidence, captured_step)
                
                gen_id = None
                confidence = 1.0 # Default confidence if not provided
                captured_step = current_step # Default to 0 lag if unknown
                
                if len(item) == 5:
                    position, delta, gen_id, confidence, captured_step = item
                elif len(item) == 4:
                    position, delta, gen_id, confidence = item
                elif len(item) == 3:
                    position, delta, gen_id = item
                else:
                    position, delta = item
                
                # STRICT SAFETY CHECK: Discard if from different generation
                if gen_id is not None and self.current_generation_id is not None:
                    if gen_id != self.current_generation_id:
                        # logger.debug(f"  [State] REJECTED delta from mismatched generation ({gen_id} != {self.current_generation_id})")
                        continue
                
                # =========================================================
                # LAG-AWARE DECAY
                # =========================================================
                if current_step is not None and captured_step is not None:
                    lag = current_step - captured_step
                    # If lag is negative (impossible if clocks consistent) or too large, prune
                    if lag > self.max_lag:
                        logger.debug(f"  [State] PRUNED delta due to high lag ({lag} > {self.max_lag})")
                        continue
                    
                    # Apply exponential decay based on lag
                    # lag 0 -> factor 1.0
                    # lag 1 -> factor 0.9
                    # lag k -> factor 0.9^k
                    lag_factor = self.delta_decay ** max(0, lag)
                else:
                    lag_factor = 1.0

                # Combine confidence scale and lag decay
                final_scale = []
                
                if isinstance(confidence, list):
                    # Vector confidence (per layer)
                    for c in confidence:
                        # Apply Sigmoid to convert value logit to probability/gate [0, 1]
                        val = 1.0 / (1.0 + math.exp(-c)) 
                        effective_alpha = self.injection_rate * val
                        final_scale.append(effective_alpha * lag_factor)
                else:
                    # Scalar confidence
                    # Assume it might be logit too if it comes from the same head
                    try:
                        c_val = float(confidence)
                        val = 1.0 / (1.0 + math.exp(-c_val))
                    except:
                        val = 0.5
                    effective_alpha = self.injection_rate * val
                    final_scale = effective_alpha * lag_factor
                
                # Buffer delta (Target State) instead of applying immediately.
                # This ensures it is applied at the exact step it was planned for.
                self._targeted_deltas[position] = (delta, final_scale)
                
                logger.debug(f"  [State] Buffered Targeted Injection for pos {position} (Lag: {lag if current_step else '?'}, Factor: {lag_factor:.2f})")
        except asyncio.QueueEmpty:
            pass

    def _compute_confidence(self, action, delta):
        """Estimate confidence in deliberated action."""
        if delta is None:
            return 0.5
        
        # Use magnitude of state change as confidence proxy
        total_magnitude = sum(
            torch.norm(d).item() for d in delta if d is not None
        )
        
        # Normalize to [0, 1]
        confidence = min(1.0, total_magnitude * 10)
        return confidence
    
    def _flush_queues(self):
        """Clear all async queues."""
        if self.state_update_queue:
            while not self.state_update_queue.empty():
                try: self.state_update_queue.get_nowait()
                except: pass

        if self._deliberation_requests:
            while not self._deliberation_requests.empty():
                try: self._deliberation_requests.get_nowait()
                except: pass
        
        self._targeted_deltas.clear()
        logger.info("✓ Flushed async queues and targeted buffer")

    async def think_and_generate(self, prompt, max_length=100, use_deliberation=True, deliberation_lag=3, fallback_to_sampling=True):
        """
        Generate tokens with asynchronous deliberation.
        """
        self._ensure_async_components()
        
        if use_deliberation:
            self._start_background_deliberation()
        
        self._flush_queues()
        self.current_generation_id = str(uuid.uuid4())
        logger.info(f"Starting generation session: {self.current_generation_id}")

        # 1. PREFILL: Process the prompt to get initial state
        # We process the prompt *without* passing existing states (since it's the start)
        # to get the state at the end of the prompt.
        with torch.no_grad():
            logits, states = self.backbone(prompt) # states=None implicit
        
        # Determine the first token to generate based on the last token of prompt
        # Actually, we need to loop for generation.
        # Initialize loop variables
        current_sequence = prompt.clone()
        self.latent_state = states # Sync for MCTS
        
        # Last token of prompt is the input for the first generation step?
        # No, 'logits' from prefill contains prediction for next token at the last position.
        # But we enter the loop to deliberate/sample.
        
        # We need to handle the first step carefully. 
        # The loop expects to generate *new* tokens.
        # But we already have the logits for the *first* new token from prefill.
        # However, the loop structure below assumes we step *into* the loop.
        
        # Let's adjust the loop to generate 'max_length' tokens.
        # We need to feed the *last generated/prompt* token to the model in each step.
        
        next_input_token = prompt[:, -1:]
        
        for step in range(max_length):
            self._current_step_index = step

            # ============================================
            # ASYNC: Request deliberation
            # ============================================
            if use_deliberation:
                future_position = step + deliberation_lag
                self._request_deliberation(future_position, current_sequence.clone())

            # ============================================
            # GENERATION (Single Step)
            # ============================================
            # In the first step (step=0), we already computed logits for prompt[:, -1] in prefill?
            # Wait, Mamba's forward(prompt) returns logits for [T1, T2, ... Tn+1].
            # The last logit corresponds to the prediction after seeing the whole prompt.
            # So for step 0, we don't need to run the model *again* if we use the prefill logits.
            # BUT, we might want to apply deliberation deltas *before* sampling.
            
            # Simplified Logic:
            # We run the model on the *last token* to get the next logits.
            # For step 0, we rely on the prefill state, but we haven't 'stepped' with the last token yet?
            # No, forward(prompt) processes *all* tokens. The state is after the *last* token.
            # So the logits for the *next* token are already available in 'logits[:, -1, :]'.
            
            # However, the loop structure below is easier if we always step.
            # Let's say we rely on the prefill logits for the first iteration?
            # Or we re-run the last token? No, that's redundant.
            
            # Let's just use the logits we have (from prefill or previous step).
            pass # Logits are ready from previous block
            
            # Check for deliberation
            if use_deliberation:
                if fallback_to_sampling:
                    suggestion = self._get_suggestion(step, timeout=0)
                else:
                    suggestion = self._get_suggestion(step, timeout=1.0)
            else:
                suggestion = None
            
            if suggestion is not None:
                next_token, confidence = suggestion
                logger.debug(f"  [MCTS] Using deliberated token {next_token}")
            else:
                # Use logits from previous step (or prefill)
                # logits shape (B, Seq, V). We want last step.
                next_token = self.sample_next_token(logits[:, -1, :])
            
            # Append to sequence
            token_tensor = torch.tensor([[next_token]], device=prompt.device)
            current_sequence = torch.cat([current_sequence, token_tensor], dim=1)
            
            # ============================================
            # STATE UPDATE (Prepare for next step)
            # ============================================
            # Now we must advance the state with this new token 'next_token'
            # This prepares 'logits' and 'states' for the NEXT iteration.
            
            async with self._state_lock:
                self.latent_state = [s.clone() if s is not None else None for s in states]
            
            await self._apply_pending_deltas(current_step=step)
            
            if step in self._targeted_deltas:
                delta, final_scale = self._targeted_deltas.pop(step)
                async with self._state_lock:
                    self.apply_state_delta(delta, scale=final_scale)

            # Sync local states
            async with self._state_lock:
                states = [s.clone() if s is not None else None for s in self.latent_state]
                
            # Run model for NEXT step (Step T -> T+1)
            with torch.no_grad():
                # CRITICAL FIX: Only pass the NEW token (next_token) and the current state.
                logits, states = self.backbone.step(token_tensor, states)

        if use_deliberation:
            self._stop_background_deliberation()
        
        return current_sequence.squeeze(0)
    
    def start_autonomous_thinking(self):
        """Start autonomous thinking loop if in autonomous mode"""
        if not self.autonomous_mode:
            return
            
        if self._thinking_task is None:
            loop = asyncio.get_event_loop()
            self._thinking_task = loop.create_task(self._autonomous_loop())
            
    async def _autonomous_loop(self):
        """Internal autonomous thinking loop"""
        while self.autonomous_mode:
            if self.latent_state is None:
                self.init_latent_state(device=next(self.parameters()).device)

            if self._state_lock is None:
                self._state_lock = asyncio.Lock()

            async with self._state_lock:
                try:
                    state_copy = [s.clone() if s is not None else None for s in self.latent_state]
                except Exception:
                    state_copy = self.latent_state

            suggestion, delta, confidence = await self.reasoning.deliberate(None, state_copy)
            if delta is not None:
                # Handle list confidence here too
                final_scale = []
                if isinstance(confidence, list):
                    for c in confidence:
                        val = 1.0 / (1.0 + math.exp(-c))
                        final_scale.append(self.injection_rate * val)
                else:
                    try:
                        c_val = float(confidence)
                        val = 1.0 / (1.0 + math.exp(-c_val))
                    except:
                        val = 0.5
                    final_scale = self.injection_rate * val
                
                async with self._state_lock:
                    self.apply_state_delta(delta, scale=final_scale)

            await asyncio.sleep(0.1)
            
    def stop_autonomous_thinking(self):
        """Stop autonomous thinking loop"""
        self.autonomous_mode = False
        if self._thinking_task is not None:
            self._thinking_task.cancel()
            self._thinking_task = None