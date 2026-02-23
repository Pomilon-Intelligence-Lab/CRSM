"""
Basic tests for CRSM model functionality.
"""

import torch
import pytest
import asyncio
from crsm.core import CRSM, CRSMConfig, CRSMModel
from crsm.core.mamba import MambaModel
from crsm.tasks.lm_task import LanguageModelingTask
from crsm.training.trainer import Trainer
from crsm.training.utils import set_seed

@pytest.fixture
def model():
    return CRSM(
        vocab_size=1000,
        d_model=128,
        d_state=32,
        d_ffn=512,
        num_layers=2
    )

def test_model_forward(model):
    # Test basic forward pass
    x = torch.randint(0, 1000, (1, 10))  # Batch size 1, sequence length 10
    logits, states = model(x)
    
    assert logits.shape == (1, 10, 1000)
    assert len(states) == 2  # Number of layers
    
@pytest.mark.asyncio
async def test_think_and_generate(model):
    # Test generation with reasoning
    prompt = torch.randint(0, 1000, (1, 5))  # Short prompt
    output = await model.think_and_generate(prompt, max_length=10)
    
    assert output.shape[0] == 15  # prompt (5) + generated (10)
    
@pytest.mark.asyncio
async def test_autonomous_mode(model):
    # Test autonomous thinking mode
    model.autonomous_mode = True
    model.start_autonomous_thinking()
    
    # Let it think for a brief moment
    await asyncio.sleep(0.5)
    
    model.stop_autonomous_thinking()
    assert model._thinking_task is None

def test_predict_policy_value():
    m = MambaModel(vocab_size=500, d_model=64, d_state=32, d_ffn=128, num_layers=2)
    x = torch.randint(0, 500, (2, 8))
    logits, value, states = m.predict_policy_value(x)
    assert logits.shape == (2, 8, 500)
    
    # Expect list of values (one per layer)
    assert isinstance(value, list)
    assert len(value) == 2
    assert value[0].shape == (2,)

def test_trainer_smoke():
    # Run a tiny training epoch to ensure trainer runs
    set_seed(0)
    config = {'batch_size': 2, 'epochs': 1, 'lr': 1e-3}
    model_config = CRSMConfig(vocab_size=256, hidden_size=64, num_hidden_layers=2)
    model = CRSMModel(model_config)
    task = LanguageModelingTask(vocab_size=256, seq_len=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, optimizer, config)
    trainer.fit(task, epochs=1)