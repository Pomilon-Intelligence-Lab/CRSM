import os
import yaml
import torch
import argparse
from crsm.core import CRSMModel, CRSMConfig, LatentDynamics
from crsm.tasks.lm_task import LanguageModelingTask
from crsm.tasks.distillation import DistillationTask
from crsm.training.trainer import Trainer
from crsm.training.logger import logger
from crsm.training.utils import set_seed

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description="CRSM Runner")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config")
    parser.add_argument('--task', type=str, default='lm', choices=['lm', 'distill', 'arc'], help="Task to run")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # 1. Load Configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(args.seed)

    # 2. Instantiate Model
    if args.task in ['lm', 'arc']:
        model_config = CRSMConfig(
            vocab_size=config.get('vocab_size', 1024),
            hidden_size=config.get('hidden_size', 256),
            num_hidden_layers=config.get('num_hidden_layers', 4),
            d_state=config.get('d_state', 64),
            intermediate_size=config.get('intermediate_size', 1024),
            injection_rate=config.get('injection_rate', 0.05)
        )
        model = CRSMModel(model_config)
        
        if args.task == 'lm':
            task = LanguageModelingTask(
                vocab_size=model_config.vocab_size,
                seq_len=config.get('seq_len', 32),
                data_dir=config.get('data_dir'),
                hf_tokenizer_name=config.get('hf_tokenizer_name')
            )
        else:
            from crsm.tasks.arc_task import ARCTask
            task = ARCTask(
                data_path=config.get('arc_data_path'),
                eval_path=config.get('arc_eval_path'),
                seq_len=config.get('seq_len', 1024)
            )
    elif args.task == 'distill':
        # For distillation, we are training the Dynamics Model
        model = LatentDynamics(
            d_model=config.get('hidden_size', 256),
            num_layers=config.get('num_hidden_layers', 4)
        )
        task = DistillationTask(shards_dir=config.get('shards_dir'))
    else:
        raise ValueError(f"Unknown task: {args.task}")

    # Log parameter count
    total_params = count_parameters(model)
    logger.info(f"Total trainable parameters: {total_params:,}")
    if total_params < 500000:
        logger.info("Target achieved: Nano-scale model detected (< 500k params)")

    # 3. Training Engine
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get('lr', 1e-3)))
    trainer = Trainer(model, optimizer, config)

    # 4. Run
    trainer.fit(task, epochs=config.get('epochs', 5))

if __name__ == "__main__":
    main()
