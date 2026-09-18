"""
Pre-Training Diagnostic Script
Checks all requirements before starting the long training run
"""

import os
import sys
import torch
import importlib.util

def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def check_gpu():
    """Check GPU availability and specifications"""
    print_section("GPU CHECK")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        print("   Training will be extremely slow on CPU")
        print("   Recommendation: Use a GPU instance")
        return False
    
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ PyTorch version: {torch.__version__}")
    
    gpu_count = torch.cuda.device_count()
    print(f"✓ Number of GPUs: {gpu_count}")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"\n  GPU {i}: {gpu_name}")
        print(f"  Memory: {gpu_memory:.2f} GB")
        
        if gpu_memory < 6.0:
            print(f"  ⚠ Warning: Low memory ({gpu_memory:.1f} GB)")
            print(f"     Recommendation: Use batch_size=2")
    
    # Test GPU operation
    try:
        test_tensor = torch.randn(1000, 1000).cuda()
        result = test_tensor @ test_tensor.t()
        print("\n✓ GPU computation test: PASSED")
        del test_tensor, result
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"\n❌ GPU computation test FAILED: {e}")
        return False
    
    return True

def check_dependencies():
    """Check required Python packages"""
    print_section("DEPENDENCIES CHECK")
    
    required = {
        'torch': '1.10.0',
        'torchvision': '0.11.0',
        'numpy': '1.19.0',
        'opencv-python': '4.5.0',
        'Pillow': '8.0.0',
        'tqdm': '4.60.0',
        'scipy': '1.7.0',
    }
    
    all_ok = True
    
    for package, min_version in required.items():
        # Handle package name variations
        import_name = package.replace('-', '_')
        if import_name == 'opencv_python':
            import_name = 'cv2'
        
        try:
            if import_name == 'cv2':
                import cv2
                version = cv2.__version__
                print(f"✓ {package}: {version}")
            else:
                mod = importlib.import_module(import_name)
                version = getattr(mod, '__version__', 'unknown')
                print(f"✓ {package}: {version}")
        except ImportError:
            print(f"❌ {package}: NOT INSTALLED")
            print(f"   Install with: pip install {package}>={min_version}")
            all_ok = False
    
    return all_ok

def check_dataset():
    """Check dataset availability and structure"""
    print_section("DATASET CHECK")
    
    data_dir = 'dataset/GOPRO'
    
    if not os.path.exists(data_dir):
        print(f"❌ Dataset directory not found: {data_dir}")
        print(f"\n   Expected structure:")
        print(f"   {data_dir}/")
        print(f"   ├── train/")
        print(f"   │   ├── blur/")
        print(f"   │   └── sharp/")
        print(f"   └── test/")
        print(f"       ├── blur/")
        print(f"       └── sharp/")
        return False
    
    print(f"✓ Dataset directory exists: {data_dir}")
    
    # Check subdirectories
    required_dirs = [
        'train/blur',
        'train/sharp',
        'test/blur',
        'test/sharp'
    ]
    
    for subdir in required_dirs:
        full_path = os.path.join(data_dir, subdir)
        if os.path.exists(full_path):
            num_files = len([f for f in os.listdir(full_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
            print(f"✓ {subdir}: {num_files} images")
        else:
            print(f"❌ {subdir}: NOT FOUND")
            return False
    
    return True

def check_pretrained_weights():
    """Check pretrained weights availability"""
    print_section("PRETRAINED WEIGHTS CHECK")
    
    weight_paths = {
        'Baseline MIMO-UNet': 'weights/MIMO-UNet.pkl',
        'Your checkpoint': 'results/MIMO-UNet-Enhanced/weights/Final.pkl',
    }
    
    found_any = False
    
    for name, path in weight_paths.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"✓ {name}: {path}")
            print(f"  Size: {size_mb:.1f} MB")
            found_any = True
        else:
            print(f"⚠ {name}: {path} (not found)")
    
    if not found_any:
        print(f"\n⚠ No pretrained weights found")
        print(f"  Training will start from scratch (takes longer)")
    
    return True  # Not critical

def check_disk_space():
    """Check available disk space"""
    print_section("DISK SPACE CHECK")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        
        total_gb = total / (1024**3)
        used_gb = used / (1024**3)
        free_gb = free / (1024**3)
        
        print(f"Total: {total_gb:.1f} GB")
        print(f"Used:  {used_gb:.1f} GB")
        print(f"Free:  {free_gb:.1f} GB")
        
        # Estimate space needed
        checkpoint_space = 0.5  # GB per checkpoint
        num_checkpoints = 15  # Save every 100 epochs for 1500 epochs
        total_needed = checkpoint_space * num_checkpoints + 5  # +5 GB buffer
        
        print(f"\nEstimated space needed: {total_needed:.1f} GB")
        
        if free_gb < total_needed:
            print(f"⚠ Warning: Low disk space!")
            print(f"  Consider cleaning up or reducing save_freq")
            return False
        else:
            print(f"✓ Sufficient disk space available")
            return True
            
    except Exception as e:
        print(f"⚠ Could not check disk space: {e}")
        return True

def check_model_import():
    """Check if model can be imported"""
    print_section("MODEL IMPORT CHECK")
    
    try:
        # Check if models directory exists
        if not os.path.exists('models'):
            print("❌ 'models' directory not found")
            print("   Make sure you're in the project root directory")
            return False
        
        # Try importing the model
        from models.MIMOUNet import build_net
        print("✓ Model import successful")
        
        # Try building the model
        try:
            model = build_net('MIMO-UNet-Enhanced')
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✓ Model build successful")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
            
            # Test forward pass
            if torch.cuda.is_available():
                model = model.cuda()
                test_input = torch.randn(1, 3, 256, 256).cuda()
                with torch.no_grad():
                    output = model(test_input)
                print(f"✓ Forward pass test: PASSED")
                del model, test_input, output
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            print(f"❌ Model build failed: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Model import failed: {e}")
        print(f"   Check that models/MIMOUNet.py exists and is correct")
        return False

def check_training_scripts():
    """Check if training scripts exist"""
    print_section("TRAINING SCRIPTS CHECK")
    
    scripts = {
        'main_improved.py': 'Main training script',
        'train_enhanced.py': 'Enhanced training functions',
        'ablation_study.py': 'Ablation study script',
    }
    
    all_exist = True
    
    for script, description in scripts.items():
        if os.path.exists(script):
            print(f"✓ {script}: {description}")
        else:
            print(f"❌ {script}: NOT FOUND")
            print(f"   {description}")
            all_exist = False
    
    return all_exist

def estimate_training_time():
    """Estimate total training time"""
    print_section("TRAINING TIME ESTIMATE")
    
    # Assumptions
    epochs = 1500
    batch_size = 4
    images_per_epoch = 2103  # GoPro train set
    
    # Time estimates (based on typical RTX 3090 performance)
    time_per_batch_ms = 800  # milliseconds
    
    batches_per_epoch = images_per_epoch // batch_size
    time_per_epoch_min = (batches_per_epoch * time_per_batch_ms) / 1000 / 60
    total_time_hours = (time_per_epoch_min * epochs) / 60
    
    print(f"Configuration:")
    print(f"  Total epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Images per epoch: {images_per_epoch}")
    print(f"  Batches per epoch: {batches_per_epoch}")
    
    print(f"\nEstimated time:")
    print(f"  Per batch: {time_per_batch_ms} ms")
    print(f"  Per epoch: {time_per_epoch_min:.1f} minutes")
    print(f"  Total: {total_time_hours:.1f} hours ({total_time_hours/24:.1f} days)")
    
    print(f"\n⚠ Note: Actual time depends on your GPU")
    print(f"  RTX 4070: ~3-4 days")
    print(f"  RTX 3090: ~2-3 days")
    print(f"  V100: ~4-5 days")

def check_output_directories():
    """Check and create output directories"""
    print_section("OUTPUT DIRECTORIES CHECK")
    
    dirs = [
        'results/',
        'results/MIMO-UNet-Enhanced/',
        'results/MIMO-UNet-Enhanced/weights/',
        'results/MIMO-UNet-Enhanced/result_image/',
    ]
    
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path} exists")
        else:
            try:
                os.makedirs(dir_path)
                print(f"✓ {dir_path} created")
            except Exception as e:
                print(f"❌ Failed to create {dir_path}: {e}")
                return False
    
    return True

def generate_training_command():
    """Generate the recommended training command"""
    print_section("RECOMMENDED TRAINING COMMAND")
    
    has_checkpoint = os.path.exists('results/MIMO-UNet-Enhanced/weights/Final.pkl')
    
    if has_checkpoint:
        print("Option A: Continue from your checkpoint (RECOMMENDED)")
        print("-" * 70)
        cmd = """python main_improved.py \\
    --model_name MIMO-UNet-Enhanced \\
    --mode train \\
    --resume results/MIMO-UNet-Enhanced/weights/Final.pkl \\
    --num_epoch 1500 \\
    --stage1_epochs 0 \\
    --learning_rate 5e-5 \\
    --use_cosine_annealing True \\
    --restart_period 100 \\
    --use_augmentation True \\
    --freq_loss_weight 0.1 \\
    --batch_size 4 \\
    --valid_freq 10 \\
    --save_freq 50"""
        print(cmd)
        print("\nExpected result: 33.5-33.8 dB PSNR")
        print("Expected time: ~3 days")
    else:
        print("Option B: Train from scratch with pretrained weights")
        print("-" * 70)
        cmd = """python main_improved.py \\
    --model_name MIMO-UNet-Enhanced \\
    --mode train \\
    --num_epoch 1500 \\
    --stage1_epochs 100 \\
    --pretrained_path weights/MIMO-UNet.pkl \\
    --learning_rate 1e-4 \\
    --use_cosine_annealing True \\
    --restart_period 100 \\
    --use_augmentation True \\
    --freq_loss_weight 0.1 \\
    --batch_size 4 \\
    --valid_freq 10 \\
    --save_freq 50"""
        print(cmd)
        print("\nExpected result: 33.5-34.0 dB PSNR")
        print("Expected time: ~5 days")
    
    print("\n" + "-" * 70)
    print("\nTo run in background:")
    print(f"nohup {cmd} > training.log 2>&1 &")
    print("\nTo monitor:")
    print("tail -f training.log")

def main():
    """Run all diagnostic checks"""
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║         PRE-TRAINING DIAGNOSTIC SCRIPT                         ║
    ║                                                                ║
    ║  Checking all requirements before starting long training run  ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    checks = {
        'GPU': check_gpu(),
        'Dependencies': check_dependencies(),
        'Dataset': check_dataset(),
        'Pretrained Weights': check_pretrained_weights(),
        'Disk Space': check_disk_space(),
        'Model Import': check_model_import(),
        'Training Scripts': check_training_scripts(),
        'Output Directories': check_output_directories(),
    }
    
    # Summary
    print_section("DIAGNOSTIC SUMMARY")
    
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    
    for name, status in checks.items():
        status_str = "✓ PASS" if status else "❌ FAIL"
        print(f"{name:<25} {status_str}")
    
    print(f"\n{passed}/{total} checks passed")
    
    if passed == total:
        print("\n✓ All checks passed! Ready to start training.")
        estimate_training_time()
        generate_training_command()
    else:
        print("\n⚠ Some checks failed. Please fix the issues before training.")
        print("\nCommon fixes:")
        print("- Install missing dependencies: pip install -r requirements.txt")
        print("- Download GoPro dataset and place in dataset/GOPRO/")
        print("- Ensure you're in the project root directory")
        print("- Copy the improved training scripts to your project")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
