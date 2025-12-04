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
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from .logger import logger

from .mamba_ssm import MambaModel
from .reasoning import AsyncDeliberationLoop
from .latent_dynamics import LatentDynamics

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
        self.dynamics = LatentDynamics(d_model=d_model)
        
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

    def apply_state_delta(self, delta, scale: Optional[float] = None):
        """
        Apply state update from deliberation.
        
        NOTE: 'delta' here is now interpreted as the 'TARGET STATE' from MCTS, 
        not a perturbation vector, when using Gated Injection.
        
        Formula: state = (1 - alpha) * state + alpha * target_state
        """
        if delta is None or self.latent_state is None:
            return
        
        # Use configured injection rate
        alpha = self.injection_rate
        if scale is not None:
            alpha = scale
            
        try:
            if isinstance(self.latent_state, list) and isinstance(delta, list):
                for i in range(min(len(self.latent_state), len(delta))):
                    if self.latent_state[i] is None or delta[i] is None:
                        continue
                    # Gated Injection: (1 - alpha) * state + alpha * target
                    self.latent_state[i] = (1 - alpha) * self.latent_state[i] + alpha * delta[i]
            else:
                # Gated Injection
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
                
                # Run MCTS (this is the slow part, but it's in background)
                logger.debug(f"  [Deliberation] Planning for position {position}...")
                
                # Run on GPU but in separate async context
                action, delta, confidence = await asyncio.to_thread(
                    self.reasoning.deliberate_sync,
                    sequence,
                    state_copy
                )
                
                # Store result
                self._deliberation_results[position] = (action, confidence)
                
                # Queue state update (Now including confidence!)
                if delta is not None:
                    await self.state_update_queue.put((position, delta, gen_id, confidence))
                
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
                
                gen_id = None
                confidence = 1.0 # Default confidence if not provided
                
                if len(item) == 4:
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
                # GATED INJECTION UPDATE (CONFIDENCE SCALED)
                # =========================================================
                
                # Scale the injection rate by MCTS confidence.
                # If MCTS is unsure (low value), we inject very little.
                # effective_alpha = base_rate * confidence
                effective_alpha = self.injection_rate * max(0.0, min(1.0, confidence))
                
                # Apply delta (Target State) to current state
                async with self._state_lock:
                    self.apply_state_delta(delta, scale=effective_alpha)
                
                logger.debug(f"  [State] Applied Gated Injection from pos {position} (Alpha: {effective_alpha:.4f}, Conf: {confidence:.2f})")
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
        logger.info("✓ Flushed async queues")

    async def think_and_generate(self, prompt, max_length=100, use_deliberation=True, deliberation_lag=3, fallback_to_sampling=True):
        """
        Generate tokens with asynchronous deliberation.
        
        Args:
            use_deliberation: Enable background MCTS
            deliberation_lag: Tokens ahead to deliberate (0=synchronous, 3=async)
            fallback_to_sampling: If True, use sampling when deliberation isn't ready
                                  If False, wait for deliberation (slower but more thoughtful)
        """
        self._ensure_async_components()
        
        # Start background deliberation task
        if use_deliberation:
            self._start_background_deliberation()
        
        # NEW: Flush queues and set generation ID
        self._flush_queues()
        self.current_generation_id = str(uuid.uuid4())
        logger.info(f"Starting generation session: {self.current_generation_id}")

        current_sequence = prompt.clone()
        states = self.backbone.init_state(batch_size=1, device=prompt.device)
        self.latent_state = states
        
        generated_tokens = []
        
        for step in range(max_length):
            # ============================================
            # ASYNC: Request deliberation (Pre-emptive)
            # ============================================
            if use_deliberation:
                # If lag is 0, we want to deliberate on THIS token immediately.
                # If lag > 0, we are pipelining for future.
                future_position = step + deliberation_lag
                self._request_deliberation(future_position, current_sequence.clone())

            # ============================================
            # FAST PATH: Generate token immediately
            # ============================================
            with torch.no_grad():
                logits, states = self.backbone(current_sequence, states)
            
            # Check if deliberation has a suggestion (non-blocking)
            if use_deliberation:
                if fallback_to_sampling:
                    # Non-blocking: use suggestion if ready, else sample
                    suggestion = self._get_suggestion(step, timeout=0)
                else:
                    # Blocking: wait for deliberation (original behavior)
                    suggestion = self._get_suggestion(step, timeout=1.0)  # Wait up to 1s
            else:
                suggestion = None
            
            if suggestion is not None:
                # Use deliberated action
                next_token, confidence = suggestion
                logger.debug(f"  [MCTS] Using deliberated token {next_token} (conf: {confidence:.2f})")
            else:
                # Fallback to sampling (instant)
                next_token = self.sample_next_token(logits[0, -1])
                logger.debug(f"  [Sample] Using sampled token {next_token}")
            
            # Update sequence
            token_tensor = torch.tensor([[next_token]], device=prompt.device)
            current_sequence = torch.cat([current_sequence, token_tensor], dim=1)
            generated_tokens.append(next_token)
            
            # ============================================
            # ASYNC: Update shared state (non-blocking)
            # ============================================
            async with self._state_lock:
                self.latent_state = [s.clone() if s is not None else None for s in states]
            
            # Check for state updates from deliberation
            await self._apply_pending_deltas(current_step=step)
            
            # Sync local states back from shared state to ensure deltas are used
            async with self._state_lock:
                states = [s.clone() if s is not None else None for s in self.latent_state]

        # Stop background task
        if use_deliberation:
            self._stop_background_deliberation()
        
        # return torch.tensor(generated_tokens, device=prompt.device)
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
                effective_alpha = self.injection_rate * max(0.0, min(1.0, confidence))
                async with self._state_lock:
                    self.apply_state_delta(delta, scale=effective_alpha)

            await asyncio.sleep(0.1)
            
    def stop_autonomous_thinking(self):
        """Stop autonomous thinking loop"""
        self.autonomous_mode = False
        if self._thinking_task is not None:
            self._thinking_task.cancel()
            self._thinking_task = None