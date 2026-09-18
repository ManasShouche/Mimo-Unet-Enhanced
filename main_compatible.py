"""
Modified main.py for enhanced training
Compatible with your existing codebase structure

USAGE:
1. Keep your original main.py as backup
2. Replace with this version OR add these arguments to your existing main.py
3. Use --use_enhanced flag to enable enhanced training
"""

import os
import torch
import argparse
from torch.backends import cudnn

# Import your existing model builder
from models.MIMOUNet import build_net

# Import both training functions
from train import _train  # Your original training
from train_enhanced import _train_enhanced  # Enhanced version

# Import your existing eval
from eval import _eval


def main(args):
    # CUDNN
    cudnn.benchmark = True

    # Create directories
    if not os.path.exists('results/'):
        os.makedirs('results/')
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # Build model
    model = build_net(args.model_name)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*70}")
    print(f"Model: {args.model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Parameter size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    if torch.cuda.is_available():
        model.cuda()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (Warning: Training will be very slow!)")
    print(f"{'='*70}\n")
    
    # Choose training mode
    if args.mode == 'train':
        # Use enhanced training if flag is set
        if hasattr(args, 'use_enhanced') and args.use_enhanced:
            print("🚀 Using ENHANCED training mode")
            _train_enhanced(model, args)
        else:
            print("📦 Using STANDARD training mode")
            _train(model, args)

    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ========================================================================
    # Basic Configuration (YOUR EXISTING ARGUMENTS)
    # ========================================================================
    parser.add_argument('--model_name', 
                       default='MIMO-UNetPlus',  # Your default
                       choices=[
                           'MIMO-UNet',
                           'MIMO-UNetPlus',
                       ], 
                       type=str,
                       help='Model architecture')
    
    parser.add_argument('--data_dir', 
                       type=str, 
                       default='dataset/GOPRO',
                       help='Path to dataset')
    
    parser.add_argument('--mode', 
                       default='train', 
                       choices=['train', 'test'], 
                       type=str)

    # ========================================================================
    # Training Hyperparameters (YOUR EXISTING ARGUMENTS)
    # ========================================================================
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=3000)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--valid_freq', type=int, default=100)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr_steps', type=list, default=[(x+1) * 500 for x in range(3000//500)])

    # ========================================================================
    # NEW: Enhanced Training Options
    # ========================================================================
    parser.add_argument('--use_enhanced',
                       action='store_true',
                       help='Use enhanced training with cosine annealing')
    
    parser.add_argument('--pretrained_path', 
                       type=str, 
                       default='',
                       help='Path to pretrained weights')
    
    parser.add_argument('--use_cosine_annealing', 
                       action='store_true',
                       help='Use cosine annealing LR schedule')
    
    parser.add_argument('--restart_period', 
                       type=int, 
                       default=100,
                       help='Restart period for cosine annealing')
    
    parser.add_argument('--freq_loss_weight', 
                       type=float, 
                       default=0.1,
                       help='Weight for frequency domain loss')
    
    parser.add_argument('--gradient_clip', 
                       type=float, 
                       default=1.0,
                       help='Gradient clipping max norm')

    # ========================================================================
    # Testing Configuration (YOUR EXISTING ARGUMENTS)
    # ========================================================================
    parser.add_argument('--test_model', 
                       type=str, 
                       default='weights/MIMO-UNetPlus.pkl')
    parser.add_argument('--save_image', 
                       type=bool, 
                       default=False, 
                       choices=[True, False])

    args = parser.parse_args()
    
    # Set output directories
    args.model_save_dir = os.path.join('results/', args.model_name, 'weights/')
    args.result_dir = os.path.join('results/', args.model_name, 'result_image/')
    
    # Print configuration
    print(f"\n{'='*70}")
    print("CONFIGURATION")
    print(f"{'='*70}")
    print(f"Model:          {args.model_name}")
    print(f"Mode:           {args.mode}")
    print(f"Data dir:       {args.data_dir}")
    
    if args.mode == 'train':
        print(f"\nTraining Parameters:")
        print(f"  Batch size:     {args.batch_size}")
        print(f"  Learning rate:  {args.learning_rate}")
        print(f"  Epochs:         {args.num_epoch}")
        print(f"  Enhanced mode:  {args.use_enhanced}")
        
        if args.use_enhanced:
            print(f"\nEnhanced Features:")
            print(f"  Cosine anneal:  {args.use_cosine_annealing}")
            if args.use_cosine_annealing:
                print(f"  Restart period: {args.restart_period}")
            print(f"  Freq loss wt:   {args.freq_loss_weight}")
            print(f"  Gradient clip:  {args.gradient_clip}")
        
        if args.pretrained_path:
            print(f"\n  Pretrained:     {args.pretrained_path}")
        if args.resume:
            print(f"  Resume from:    {args.resume}")
    
    print(f"{'='*70}\n")
    
    main(args)
