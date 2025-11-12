"""Command-line entry points for CRSM experiments."""
import argparse
from .train import main as train_main


def main():
    parser = argparse.ArgumentParser(prog='crsm')
    sub = parser.add_subparsers(dest='cmd')

    train_p = sub.add_parser('train')
    train_p.add_argument('--epochs', type=int, default=1)
    train_p.add_argument('--batch-size', type=int, default=8)
    train_p.add_argument('--vocab-size', type=int, default=1000)
    train_p.add_argument('--seq-len', type=int, default=32)
    train_p.add_argument('--lr', type=float, default=1e-3)
    train_p.add_argument('--seed', type=int, default=42)
    train_p.add_argument('--data-dir', type=str, default=None)
    train_p.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    train_p.add_argument('--device', type=str, default=None)
    train_p.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    train_p.add_argument('--wandb-project', type=str, default='crsm')
    train_p.add_argument('--grad-accum', type=int, default=1, help='Gradient accumulation steps')
    train_p.add_argument('--amp', action='store_true', help='Enable automatic mixed precision (CUDA only)')
    train_p.add_argument('--num-workers', type=int, default=0, help='DataLoader num_workers')
    train_p.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    train_p.add_argument('--distributed', action='store_true', help='Enable single-node multi-GPU distributed training')
    train_p.add_argument('--local-rank', type=int, default=None, help='Local rank supplied by launcher')
    train_p.add_argument('--no-value-loss', action='store_true', help='Disable value head training')
    train_p.add_argument('--tokenizer', type=str, default=None, help='HuggingFace tokenizer name (e.g., gpt2)')

    args = parser.parse_args()
    if args.cmd == 'train':
        train_main(
            epochs=args.epochs,
            batch_size=args.batch_size,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            lr=args.lr,
            seed=args.seed,
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            device_str=args.device,
            wandb_enabled=(not args.no_wandb),
            wandb_project=args.wandb_project,
            grad_accum_steps=args.grad_accum,
            use_amp=args.amp,
            num_workers=args.num_workers,
            resume=args.resume,
            distributed=args.distributed,
            local_rank=args.local_rank,
            use_value_loss=(not args.no_value_loss),
            hf_tokenizer_name=args.tokenizer
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()