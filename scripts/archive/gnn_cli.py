#!/usr/bin/env python3
"""CLI interface for GNN training and evaluation."""

import argparse
import subprocess
import sys
from pathlib import Path


def run_training(args):
    """Run GNN training."""
    if args.ensemble:
        # Use ensemble training script
        cmd = [
            "python3", "scripts/train_gnn_ensemble.py",
            "+preset=bulk",
            f"dataset.batch_size={args.batch_size}",
            f"gnn.num_epochs={args.epochs}",
            f"gnn.learning_rate={args.learning_rate}",
            f"+w_f={args.w_f}",
            f"+w_e={args.w_e}",
            f"+w_s={args.w_s}",
            f"+scheduler_type={args.scheduler}",
            f"+early_stopping_patience={args.patience}",
            f"+max_grad_norm={args.max_grad_norm}",
            f"+ensemble={args.ensemble}"
        ]
        
        print(f"🎯 Starting ensemble GNN training ({args.ensemble} models)...")
    else:
        # Use single model training script
        cmd = [
            "python3", "scripts/train_gnn_optimized.py",
            "+preset=bulk",
            f"dataset.batch_size={args.batch_size}",
            f"gnn.num_epochs={args.epochs}",
            f"gnn.learning_rate={args.learning_rate}",
            f"+w_f={args.w_f}",
            f"+w_e={args.w_e}",
            f"+w_s={args.w_s}",
            f"+scheduler_type={args.scheduler}",
            f"+early_stopping_patience={args.patience}",
            f"+max_grad_norm={args.max_grad_norm}"
        ]
        
        print("🚀 Starting single GNN training...")
    
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


def run_evaluation(args):
    """Run GNN evaluation."""
    cmd = [
        "python3", "scripts/eval_gnn.py",
        "--checkpoint", args.checkpoint,
        "--test-size", str(args.test_size),
        "--batch-size", str(args.batch_size),
        "--device", args.device
    ]
    
    if args.save_artifacts:
        cmd.append("--save-artifacts")
    
    print("🔍 Starting GNN evaluation...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="GNN Training and Evaluation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Training subcommand
    train_parser = subparsers.add_parser("train", help="Train GNN model")
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    train_parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--w-f", type=float, default=100, help="Force loss weight")
    train_parser.add_argument("--w-e", type=float, default=1, help="Energy loss weight")
    train_parser.add_argument("--w-s", type=float, default=10, help="Stress loss weight")
    train_parser.add_argument("--scheduler", type=str, default="plateau", 
                            choices=["plateau", "cosine", "step"], help="LR scheduler")
    train_parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm")
    train_parser.add_argument("--ensemble", type=int, default=None, help="Train ensemble of N models")
    
    # Evaluation subcommand
    eval_parser = subparsers.add_parser("eval", help="Evaluate GNN model")
    eval_parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    eval_parser.add_argument("--test-size", type=int, default=5000, help="Test split size")
    eval_parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    eval_parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    eval_parser.add_argument("--save-artifacts", action="store_true", 
                           help="Save best checkpoint to artifacts/gnn/best.ckpt")
    
    args = parser.parse_args()
    
    if args.command == "train":
        return run_training(args)
    elif args.command == "eval":
        return run_evaluation(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
