"""
Enhanced Training Script - Compatible with existing Mimo-Unet-Enhanced codebase
Adds: Cosine Annealing + Better monitoring
Works seamlessly with your existing train.py and main.py
"""

import os
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from data import train_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid import _valid


def _train_enhanced(model, args):
    """
    Enhanced training with cosine annealing and better logging
    Drop-in replacement for _train() function
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pretrained weights if provided
    if hasattr(args, 'pretrained_path') and args.pretrained_path and os.path.exists(args.pretrained_path):
        print("\n" + "="*70)
        print("LOADING PRETRAINED WEIGHTS")
        print("="*70)
        checkpoint = torch.load(args.pretrained_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load compatible weights
        model_dict = model.state_dict()
        compatible_dict = {k: v for k, v in state_dict.items() 
                          if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(compatible_dict)
        model.load_state_dict(model_dict)
        
        print(f"✓ Loaded {len(compatible_dict)}/{len(state_dict)} parameters")
        print("="*70 + "\n")
    
    # Setup optimizer and loss
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Setup dataloader
    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker)
    max_iter = len(dataloader)
    
    # ENHANCED: Use cosine annealing scheduler if enabled
    if hasattr(args, 'use_cosine_annealing') and args.use_cosine_annealing:
        restart_period = getattr(args, 'restart_period', 100)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=restart_period,
            T_mult=1,
            eta_min=1e-7
        )
        print(f"✓ Using Cosine Annealing with restarts every {restart_period} epochs")
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            args.lr_steps, 
            args.gamma
        )
        print(f"✓ Using Step LR schedule")
    
    # Resume from checkpoint if specified
    epoch = 1
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        epoch = state.get('epoch', 1)
        
        if 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])
        if 'scheduler' in state:
            scheduler.load_state_dict(state['scheduler'])
        if 'model' in state:
            model.load_state_dict(state['model'])
        
        print(f'✓ Resumed from epoch {epoch}')
        epoch += 1
    
    # Setup tensorboard
    writer = SummaryWriter()
    
    # Setup metrics
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    best_psnr = -1
    
    # Get frequency loss weight
    freq_weight = getattr(args, 'freq_loss_weight', 0.1)
    
    print(f"\n{'='*70}")
    print(f"STARTING ENHANCED TRAINING")
    print(f"{'='*70}")
    print(f"Epochs: {epoch} → {args.num_epoch}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Frequency loss weight: {freq_weight}")
    print(f"{'='*70}\n")
    
    # Training loop
    for epoch_idx in range(epoch, args.num_epoch + 1):
        model.train()
        epoch_timer.tic()
        iter_timer.tic()
        
        for iter_idx, batch_data in enumerate(dataloader):
            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_img = model(input_img)
            
            # Multi-scale ground truth
            label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear', align_corners=False)
            label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear', align_corners=False)
            
            # Spatial loss (L1)
            l1 = criterion(pred_img[0], label_img4)
            l2 = criterion(pred_img[1], label_img2)
            l3 = criterion(pred_img[2], label_img)
            loss_content = l1 + l2 + l3
            
            # Frequency loss (FFT-based)
            try:
                label_fft1 = torch.fft.rfft2(label_img4, norm='ortho')
                pred_fft1 = torch.fft.rfft2(pred_img[0], norm='ortho')
                label_fft2 = torch.fft.rfft2(label_img2, norm='ortho')
                pred_fft2 = torch.fft.rfft2(pred_img[1], norm='ortho')
                label_fft3 = torch.fft.rfft2(label_img, norm='ortho')
                pred_fft3 = torch.fft.rfft2(pred_img[2], norm='ortho')
                
                f1 = criterion(pred_fft1.real, label_fft1.real) + criterion(pred_fft1.imag, label_fft1.imag)
                f2 = criterion(pred_fft2.real, label_fft2.real) + criterion(pred_fft2.imag, label_fft2.imag)
                f3 = criterion(pred_fft3.real, label_fft3.real) + criterion(pred_fft3.imag, label_fft3.imag)
                loss_fft = f1 + f2 + f3
            except:
                loss_fft = torch.tensor(0.0).to(device)
            
            # Total loss
            loss = loss_content + freq_weight * loss_fft
            loss.backward()
            
            # Gradient clipping
            if hasattr(args, 'gradient_clip') and args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            
            optimizer.step()
            
            # Track metrics
            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())
            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())
            
            # Print progress
            if (iter_idx + 1) % args.print_freq == 0:
                lr = check_lr(optimizer)
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, lr, 
                    iter_pixel_adder.average(), iter_fft_adder.average()))
                
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1) * max_iter)
                writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx-1) * max_iter)
                
                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_fft_adder.reset()
        
        # Save checkpoint
        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch_idx
        }, overwrite_name)
        
        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch_idx
            }, save_name)
        
        # Print epoch summary
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average()))
        
        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        
        # Step scheduler
        scheduler.step()
        
        # Validation
        if epoch_idx % args.valid_freq == 0:
            val_gopro = _valid(model, args, epoch_idx)
            print('%03d epoch \n Average GOPRO PSNR %.2f dB' % (epoch_idx, val_gopro))
            writer.add_scalar('PSNR_GOPRO', val_gopro, epoch_idx)
            
            if val_gopro >= best_psnr:
                best_psnr = val_gopro
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))
                print(f'✓ New best model saved! PSNR: {best_psnr:.2f} dB')
                
                gain = best_psnr - 31.73
                if gain >= 1.5:
                    print(f'🎯 TARGET REACHED! Gain: +{gain:.2f} dB')
    
    # Save final model
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)
    
    print(f"\n{'='*70}")
    print(f'TRAINING COMPLETE! Best PSNR: {best_psnr:.2f} dB (+{best_psnr - 31.73:.2f} dB)')
    print(f"{'='*70}\n")
