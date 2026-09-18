"""
Training Progress Monitor
Real-time tracking of PSNR, loss, and estimated completion time
"""

import re
import time
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class TrainingMonitor:
    """
    Parse training logs and generate progress reports
    """
    
    def __init__(self, log_file='training.log'):
        self.log_file = log_file
        self.epochs = []
        self.psnr_values = []
        self.train_losses = []
        self.pixel_losses = []
        self.freq_losses = []
        self.start_time = None
        
    def parse_log(self):
        """Parse training log file"""
        if not Path(self.log_file).exists():
            print(f"❌ Log file not found: {self.log_file}")
            return False
        
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            # Parse PSNR: "Val PSNR = 32.45 dB"
            psnr_match = re.search(r'PSNR[:\s=]+(\d+\.\d+)', line, re.IGNORECASE)
            if psnr_match:
                psnr = float(psnr_match.group(1))
                if psnr > 20 and psnr < 50:  # Sanity check
                    self.psnr_values.append(psnr)
            
            # Parse epoch: "Epoch 123:"
            epoch_match = re.search(r'Epoch[:\s]+(\d+)', line, re.IGNORECASE)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                self.epochs.append(epoch)
            
            # Parse train loss: "Train Loss = 0.0345"
            loss_match = re.search(r'Loss[:\s=]+(\d+\.\d+)', line, re.IGNORECASE)
            if loss_match:
                loss = float(loss_match.group(1))
                if loss < 1.0:  # Sanity check
                    self.train_losses.append(loss)
            
            # Parse pixel loss: "Pixel Loss: 0.0345"
            pixel_match = re.search(r'Pixel Loss[:\s]+(\d+\.\d+)', line, re.IGNORECASE)
            if pixel_match:
                self.pixel_losses.append(float(pixel_match.group(1)))
            
            # Parse FFT loss: "FFT Loss: 0.0512"
            freq_match = re.search(r'FFT Loss[:\s]+(\d+\.\d+)', line, re.IGNORECASE)
            if freq_match:
                self.freq_losses.append(float(freq_match.group(1)))
        
        # Align arrays (might have different lengths)
        min_len = min(len(self.epochs), len(self.psnr_values))
        self.epochs = self.epochs[:min_len]
        self.psnr_values = self.psnr_values[:min_len]
        
        return len(self.epochs) > 0
    
    def estimate_completion(self, target_epoch=1500):
        """Estimate training completion time"""
        if len(self.epochs) < 2:
            return None
        
        # Calculate time per epoch
        current_epoch = self.epochs[-1]
        elapsed_epochs = current_epoch - self.epochs[0]
        
        if elapsed_epochs == 0:
            return None
        
        # Estimate from file modification time
        file_stat = Path(self.log_file).stat()
        file_age_seconds = time.time() - file_stat.st_mtime
        time_per_epoch = file_age_seconds / elapsed_epochs
        
        remaining_epochs = target_epoch - current_epoch
        estimated_seconds = remaining_epochs * time_per_epoch
        
        return {
            'current_epoch': current_epoch,
            'target_epoch': target_epoch,
            'remaining_epochs': remaining_epochs,
            'estimated_time': timedelta(seconds=estimated_seconds),
            'completion_time': datetime.now() + timedelta(seconds=estimated_seconds),
            'time_per_epoch': timedelta(seconds=time_per_epoch)
        }
    
    def predict_final_psnr(self):
        """Predict final PSNR using linear regression on recent data"""
        if len(self.psnr_values) < 10:
            return None
        
        # Use last 50 epochs for trend prediction
        recent_epochs = np.array(self.epochs[-50:])
        recent_psnr = np.array(self.psnr_values[-50:])
        
        # Linear fit
        coeffs = np.polyfit(recent_epochs, recent_psnr, 1)
        slope, intercept = coeffs
        
        # Predict at epoch 1500
        predicted_1500 = slope * 1500 + intercept
        
        return {
            'current_psnr': self.psnr_values[-1],
            'slope': slope,
            'predicted_1500': predicted_1500,
            'gain_from_current': predicted_1500 - self.psnr_values[-1]
        }
    
    def generate_plots(self, output_dir='results/MIMO-UNet-Enhanced/'):
        """Generate training progress plots"""
        if len(self.epochs) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: PSNR over epochs
        ax = axes[0, 0]
        if len(self.psnr_values) > 0:
            ax.plot(self.epochs, self.psnr_values, 'b-', linewidth=2)
            ax.axhline(y=31.73, color='r', linestyle='--', label='Baseline (31.73 dB)')
            ax.axhline(y=33.5, color='g', linestyle='--', label='Target (33.5 dB)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('PSNR (dB)')
            ax.set_title('Validation PSNR Progress')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Training loss over epochs
        ax = axes[0, 1]
        if len(self.train_losses) > 0:
            ax.plot(self.train_losses, 'r-', linewidth=1)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.grid(True, alpha=0.3)
        
        # Plot 3: PSNR improvement rate
        ax = axes[1, 0]
        if len(self.psnr_values) > 10:
            # Calculate improvement per 10 epochs
            window = 10
            improvements = []
            epoch_centers = []
            for i in range(window, len(self.psnr_values)):
                improvement = self.psnr_values[i] - self.psnr_values[i-window]
                improvements.append(improvement)
                epoch_centers.append(self.epochs[i])
            
            ax.plot(epoch_centers, improvements, 'g-', linewidth=2)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(f'PSNR Gain per {window} epochs (dB)')
            ax.set_title('Learning Rate (PSNR improvement)')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Predicted trajectory
        ax = axes[1, 1]
        if len(self.psnr_values) > 10:
            # Plot actual
            ax.plot(self.epochs, self.psnr_values, 'b-', linewidth=2, label='Actual')
            
            # Fit and predict
            recent_epochs = np.array(self.epochs[-50:])
            recent_psnr = np.array(self.psnr_values[-50:])
            coeffs = np.polyfit(recent_epochs, recent_psnr, 1)
            
            # Extrapolate to epoch 1500
            future_epochs = np.array([self.epochs[-1], 1500])
            predicted = np.polyval(coeffs, future_epochs)
            
            ax.plot(future_epochs, predicted, 'r--', linewidth=2, label='Predicted')
            ax.axhline(y=33.5, color='g', linestyle='--', alpha=0.5, label='Target (33.5 dB)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('PSNR (dB)')
            ax.set_title('PSNR Prediction to Epoch 1500')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir) / 'training_progress.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Progress plot saved to: {output_path}")
        plt.close()
    
    def print_report(self):
        """Print comprehensive progress report"""
        print("\n" + "="*70)
        print("TRAINING PROGRESS REPORT")
        print("="*70 + "\n")
        
        if len(self.epochs) == 0:
            print("❌ No training data found in log file")
            return
        
        # Current status
        print("📊 CURRENT STATUS")
        print("-" * 70)
        print(f"Current epoch:     {self.epochs[-1]}")
        if len(self.psnr_values) > 0:
            print(f"Latest PSNR:       {self.psnr_values[-1]:.2f} dB")
            print(f"Baseline PSNR:     31.73 dB")
            print(f"Current gain:      +{self.psnr_values[-1] - 31.73:.2f} dB")
        if len(self.train_losses) > 0:
            print(f"Latest train loss: {self.train_losses[-1]:.4f}")
        
        # Progress
        print("\n📈 PROGRESS")
        print("-" * 70)
        if len(self.psnr_values) > 1:
            improvement = self.psnr_values[-1] - self.psnr_values[0]
            print(f"PSNR improvement:  +{improvement:.2f} dB")
            print(f"From epoch {self.epochs[0]} to {self.epochs[-1]}")
            
            # Last 10 epochs improvement
            if len(self.psnr_values) >= 10:
                recent_improvement = self.psnr_values[-1] - self.psnr_values[-10]
                print(f"Last 10 epochs:    +{recent_improvement:.3f} dB")
        
        # Time estimation
        est = self.estimate_completion()
        if est:
            print("\n⏱️  TIME ESTIMATION")
            print("-" * 70)
            print(f"Time per epoch:    {est['time_per_epoch']}")
            print(f"Remaining epochs:  {est['remaining_epochs']}")
            print(f"Estimated time:    {est['estimated_time']}")
            print(f"Completion ETA:    {est['completion_time'].strftime('%Y-%m-%d %H:%M')}")
        
        # PSNR prediction
        pred = self.predict_final_psnr()
        if pred:
            print("\n🎯 PSNR PREDICTION")
            print("-" * 70)
            print(f"Current PSNR:      {pred['current_psnr']:.2f} dB")
            print(f"Trend (dB/epoch):  {pred['slope']:.4f}")
            print(f"Predicted @ 1500:  {pred['predicted_1500']:.2f} dB")
            print(f"Expected gain:     +{pred['gain_from_current']:.2f} dB")
            
            # Check if target will be reached
            if pred['predicted_1500'] >= 33.5:
                print(f"\n✓ ON TRACK to reach 33.5 dB target!")
            elif pred['predicted_1500'] >= 33.0:
                print(f"\n⚠ Might need to train longer to reach 33.5 dB")
            else:
                print(f"\n⚠ Current trend suggests < 33.0 dB")
                print(f"  Consider:")
                print(f"  - Reducing learning rate")
                print(f"  - Increasing frequency loss weight")
                print(f"  - Training beyond 1500 epochs")
        
        # Milestones
        print("\n🏁 MILESTONES")
        print("-" * 70)
        milestones = {
            32.0: "Good progress",
            32.5: "Strong improvement",
            33.0: "Excellent result",
            33.5: "Publication-ready!",
            34.0: "Outstanding!"
        }
        
        current_psnr = self.psnr_values[-1] if self.psnr_values else 0
        for threshold, message in milestones.items():
            if current_psnr >= threshold:
                print(f"✓ {threshold:.1f} dB: {message}")
            else:
                print(f"  {threshold:.1f} dB: {message}")
        
        print("\n" + "="*70 + "\n")

def monitor_continuously(log_file='training.log', interval=300):
    """
    Continuously monitor training and update progress
    
    Args:
        log_file: Path to training log
        interval: Update interval in seconds (default 5 minutes)
    """
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║           TRAINING MONITOR - CONTINUOUS MODE                   ║
    ║                                                                ║
    ║  Monitoring training progress every 5 minutes                  ║
    ║  Press Ctrl+C to stop                                          ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        while True:
            monitor = TrainingMonitor(log_file)
            if monitor.parse_log():
                monitor.print_report()
                monitor.generate_plots()
            else:
                print(f"⏳ Waiting for training data in {log_file}...")
            
            print(f"\n⏸️  Next update in {interval//60} minutes...")
            print(f"   (Press Ctrl+C to stop monitoring)")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n✓ Monitoring stopped by user")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Training Progress Monitor')
    parser.add_argument('--log', 
                       type=str, 
                       default='training.log',
                       help='Path to training log file')
    parser.add_argument('--continuous', 
                       action='store_true',
                       help='Continuously monitor (update every 5 min)')
    parser.add_argument('--interval', 
                       type=int, 
                       default=300,
                       help='Update interval in seconds (continuous mode)')
    
    args = parser.parse_args()
    
    if args.continuous:
        monitor_continuously(args.log, args.interval)
    else:
        # Single snapshot
        monitor = TrainingMonitor(args.log)
        if monitor.parse_log():
            monitor.print_report()
            monitor.generate_plots()
        else:
            print(f"❌ Could not parse log file: {args.log}")
            print(f"   Make sure training has started and log file exists")

if __name__ == '__main__':
    main()
