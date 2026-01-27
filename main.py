import os
import torch
import argparse
from torch.backends import cudnn
from models.MIMOUNet import build_net
from train import _train
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
    # MODIFIED: Add new model choices
    parser.add_argument('--model_name', 
                       default='MIMO-UNet', 
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
    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)

    # Train
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

    # NEW: Add these arguments for enhanced model training
    parser.add_argument('--pretrained_path', 
                       type=str, 
                       default='',
                       help='Path to pretrained MIMO-UNet weights (.pkl file)')
    parser.add_argument('--staged_training', 
                       type=bool, 
                       default=True,
                       help='Use staged training (freeze encoder first, then fine-tune)')
    parser.add_argument('--stage1_epochs', 
                       type=int, 
                       default=50,
                       help='Number of epochs for stage 1 (training new modules only)')

    # Test
    parser.add_argument('--test_model', type=str, default='weights/MIMO-UNet.pkl')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', args.model_name, 'weights/')
    args.result_dir = os.path.join('results/', args.model_name, 'result_image/')
    
    # NEW: Print configuration
    print(f"\n{'='*60}")
    print("CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model:          {args.model_name}")
    print(f"Mode:           {args.mode}")
    print(f"Data dir:       {args.data_dir}")
    if args.mode == 'train':
        print(f"Batch size:     {args.batch_size}")
        print(f"Learning rate:  {args.learning_rate}")
        print(f"Epochs:         {args.num_epoch}")
        if args.pretrained_path:
            print(f"Pretrained:     {args.pretrained_path}")
            print(f"Staged train:   {args.staged_training}")
            if args.staged_training:
                print(f"Stage 1 epochs: {args.stage1_epochs}")
    print(f"{'='*60}\n")
    
    main(args)