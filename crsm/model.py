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
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

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
                 top_p: float = 0.95):
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
        
        self.autonomous_mode = autonomous_mode
        self._thinking_task = None
        # Background suggestion infrastructure
        self._bg_task = None
        # create queue and lock eagerly to avoid races
        self._bg_queue = asyncio.Queue(maxsize=8)

        # Canonical latent state and lock for concurrency
        self.latent_state = None
        self._state_lock = asyncio.Lock()
    
    def load_dynamics(self, dynamics_path: str):
        """Load trained dynamics model."""
        device = next(self.parameters()).device
        ckpt = torch.load(dynamics_path, map_location=device)
        
        if 'dynamics_state' in ckpt:
            self.dynamics.load_state_dict(ckpt['dynamics_state'])
            print(f"✓ Loaded dynamics from {dynamics_path}")
        elif isinstance(ckpt, dict) and all(k.startswith('net.') for k in ckpt.keys()):
            self.dynamics.load_state_dict(ckpt)
            print(f"✓ Loaded dynamics from {dynamics_path}")
        else:
            print(f"⚠ Warning: Could not load dynamics from {dynamics_path}")
            return False
        
        # Update reasoning module reference
        self.reasoning.dynamics_model = self.dynamics
        print("✓ Connected dynamics to reasoning module")
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

    def apply_state_delta(self, delta):
        """Apply a state delta produced by the deliberator to the canonical latent_state."""
        if delta is None or self.latent_state is None:
            return
        try:
            if isinstance(self.latent_state, list) and isinstance(delta, list):
                for i in range(min(len(self.latent_state), len(delta))):
                    if self.latent_state[i] is None or delta[i] is None:
                        continue
                    self.latent_state[i] = self.latent_state[i] + delta[i]
            else:
                self.latent_state = self.latent_state + delta
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
    
    async def think_and_generate(self, 
                               prompt: torch.Tensor,
                               max_length: int = 100,
                               use_sampling: bool = True) -> torch.Tensor:
        """
        Generate text with active reasoning
        Args:
            prompt: Input prompt tensor
            max_length: Maximum length to generate
            use_sampling: If True, use temperature sampling; if False, use MCTS
        Returns:
            Generated sequence tensor
        """
        current_sequence = prompt.clone()
        states = None
        
        # Background worker that continuously updates a suggested token
        async def _bg_worker():
            if self._bg_queue is None:
                self._bg_queue = asyncio.Queue(maxsize=8)
            if self._state_lock is None:
                self._state_lock = asyncio.Lock()

            try:
                while True:
                    async with self._state_lock:
                        seq_copy = current_sequence.clone()
                        state_copy = None
                        if self.latent_state is not None:
                            try:
                                state_copy = [s.clone() if s is not None else None for s in self.latent_state]
                            except Exception:
                                state_copy = self.latent_state

                    suggestion, delta = await self.reasoning.deliberate(seq_copy, state_copy)

                    try:
                        self._bg_queue.put_nowait((suggestion, delta))
                    except asyncio.QueueFull:
                        pass

                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                return

        # Start background worker only if using MCTS
        if not use_sampling:
            self._bg_task = asyncio.create_task(_bg_worker())

        for _ in range(max_length):
            # Get current logits and states
            logits, states = self.forward(current_sequence, states)

            # Update canonical latent_state
            if states is not None:
                try:
                    async with self._state_lock:
                        self.latent_state = [s.clone() if s is not None else None for s in states]
                except Exception:
                    async with self._state_lock:
                        self.latent_state = states

            if use_sampling:
                # Use temperature sampling (faster, more diverse)
                next_token = self.sample_next_token(logits[0, -1])
            else:
                # Use MCTS (slower, more deliberate)
                suggestion = None
                delta = None
                if self._bg_queue is not None:
                    try:
                        suggestion, delta = self._bg_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        suggestion = None

                if suggestion is None:
                    suggestion, delta = await self.reasoning.deliberate(current_sequence, self.latent_state)

                next_token = int(suggestion)

                if delta is not None:
                    async with self._state_lock:
                        self.apply_state_delta(delta)

            # Append token
            token_tensor = torch.tensor([[next_token]], device=prompt.device, dtype=prompt.dtype)
            current_sequence = torch.cat([current_sequence, token_tensor], dim=1)

        # Stop background worker
        if self._bg_task is not None:
            self._bg_task.cancel()
            try:
                await self._bg_task
            except asyncio.CancelledError:
                pass
            self._bg_task = None
        
        if self._bg_queue is not None:
            self._bg_queue = None

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

            suggestion, delta = await self.reasoning.deliberate(None, state_copy)
            if delta is not None:
                async with self._state_lock:
                    self.apply_state_delta(delta)

            await asyncio.sleep(0.1)
            
    def stop_autonomous_thinking(self):
        """Stop autonomous thinking loop"""
        self.autonomous_mode = False
        if self._thinking_task is not None:
            self._thinking_task.cancel()
            self._thinking_task = None