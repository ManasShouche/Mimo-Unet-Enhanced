import os
import torch
import argparse
from torch.backends import cudnn
from models.MIMOUNet import build_net

# MODIFIED: Import enhanced training function
from train_enhanced import _train_enhanced as _train

from eval import _eval


def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net(args.model_name)
    
    # NEW: Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Total parameters: {total_params:,}")
    
    if torch.cuda.is_available():
        model.cuda()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (Warning: Training will be very slow!)")
    print(f"{'='*60}\n")
    
    if args.mode == 'train':
        _train(model, args)

    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', 
                       default='MIMO-UNetPlus',  # Changed default to Plus
                       choices=[
                           'MIMO-UNet',              # Original baseline
                           'MIMO-UNetPlus',          # Original Plus
                           'MIMO-UNet-Enhanced',     # Full enhanced model
                           'MIMO-UNet-Attention',    # Only attention
                           'MIMO-UNet-Frequency',    # Only frequency
                           'MIMO-UNet-Full'          # Same as Enhanced
                       ], 
                       type=str,
                       help='Model architecture to use')
    
    parser.add_argument('--data_dir', type=str, default='dataset/GOPRO')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)

    # Train
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=1500)  # Changed from 3000 to 1500
    parser.add_argument('--print_freq', type=int, default=50)  # Changed to 50 for more frequent updates
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=50)  # Changed to 50
    parser.add_argument('--valid_freq', type=int, default=10)  # Changed to 10 for more frequent validation
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr_steps', type=list, default=[(x+1) * 500 for x in range(3000//500)])

    # Pretrained/Staged training
    parser.add_argument('--pretrained_path', 
                       type=str, 
                       default='',
                       help='Path to pretrained MIMO-UNet weights (.pkl file)')
    parser.add_argument('--staged_training', 
                       type=bool, 
                       default=False,  # Changed to False (we're using enhanced training)
                       help='Use staged training (freeze encoder first, then fine-tune)')
    parser.add_argument('--stage1_epochs', 
                       type=int, 
                       default=50,
                       help='Number of epochs for stage 1 (training new modules only)')

    # NEW: Enhanced training arguments
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

    # Test
    parser.add_argument('--test_model', type=str, default='weights/MIMO-UNet.pkl')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', args.model_name, 'weights/')
    args.result_dir = os.path.join('results/', args.model_name, 'result_image/')
    
    # NEW: Print configuration
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
        print(f"  Cosine anneal:  {args.use_cosine_annealing}")
        if args.use_cosine_annealing:
            print(f"  Restart period: {args.restart_period}")
        print(f"  Freq loss wt:   {args.freq_loss_weight}")
        print(f"  Gradient clip:  {args.gradient_clip}")
        if args.pretrained_path:
            print(f"  Pretrained:     {args.pretrained_path}")
        if args.resume:
            print(f"  Resume from:    {args.resume}")
    print(f"{'='*70}\n")
    
    main(args)