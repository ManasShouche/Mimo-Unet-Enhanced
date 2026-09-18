"""
GoPro Dataset Preprocessor for Kaggle Version
Works with blur/images/ and sharp/images/ structure
Downsamples images for faster training
"""

import os
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np


class KaggleGoPROPreprocessor:
    """
    Preprocesses Kaggle GoPro dataset:
    - Handles blur/images/ and sharp/images/ structure
    - Downsamples and splits into train/val/test
    """
    
    def __init__(self, source_dir, output_dir, downsample_factor=4, num_workers=8):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.downsample_factor = downsample_factor
        self.num_workers = num_workers
        
    def process_dataset(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Main processing function
        
        Args:
            train_ratio: Fraction for training (default: 0.8)
            val_ratio: Fraction for validation (default: 0.1)
            test_ratio: Fraction for testing (default: 0.1)
        """
        print(f"\n{'='*70}")
        print("PROCESSING KAGGLE GOPRO DATASET")
        print(f"{'='*70}\n")
        
        # Find blur and sharp directories (handle both structures)
        possible_blur_paths = [
            self.source_dir / 'blur' / 'images',  # New structure
            self.source_dir / 'blur',              # Flat structure
        ]
        possible_sharp_paths = [
            self.source_dir / 'sharp' / 'images',  # New structure
            self.source_dir / 'sharp',              # Flat structure
        ]
        
        blur_dir = None
        sharp_dir = None
        
        for path in possible_blur_paths:
            if path.exists() and list(path.glob('*.png')):
                blur_dir = path
                break
        
        for path in possible_sharp_paths:
            if path.exists() and list(path.glob('*.png')):
                sharp_dir = path
                break
        
        if not blur_dir or not sharp_dir:
            print(f"❌ Could not find image directories!")
            print(f"   Looked in:")
            print(f"   - {self.source_dir / 'blur' / 'images'}")
            print(f"   - {self.source_dir / 'blur'}")
            print(f"   - {self.source_dir / 'sharp' / 'images'}")
            print(f"   - {self.source_dir / 'sharp'}")
            raise ValueError(f"Expected PNG images in blur/images/ or blur/")
        
        print(f"✅ Found images:")
        print(f"   Blur:  {blur_dir}")
        print(f"   Sharp: {sharp_dir}\n")
        
        # Get all image pairs
        blur_images = sorted(list(blur_dir.glob('*.png')))
        
        if len(blur_images) == 0:
            raise ValueError(f"No PNG images found in {blur_dir}")
        
        # Verify all pairs exist
        image_pairs = []
        missing_count = 0
        for blur_img in blur_images:
            sharp_img = sharp_dir / blur_img.name
            if sharp_img.exists():
                image_pairs.append((blur_img, sharp_img, blur_img.name))
            else:
                missing_count += 1
                if missing_count <= 5:  # Show first 5 missing
                    print(f"⚠️  Missing sharp pair for: {blur_img.name}")
        
        if missing_count > 5:
            print(f"⚠️  ... and {missing_count - 5} more missing pairs")
        
        total_pairs = len(image_pairs)
        print(f"\n✅ Found {total_pairs} valid image pairs")
        print(f"   Downsample factor: {self.downsample_factor}x")
        
        # Get sample image info
        sample_img = Image.open(image_pairs[0][0])
        orig_w, orig_h = sample_img.size
        new_w = (orig_w // self.downsample_factor // 8) * 8  # Align to 8
        new_h = (orig_h // self.downsample_factor // 8) * 8
        
        print(f"   Original size: {orig_w}x{orig_h}")
        print(f"   New size: {new_w}x{new_h}")
        print(f"   Workers: {self.num_workers}\n")
        
        # Split dataset
        np.random.seed(42)
        indices = np.random.permutation(total_pairs)
        
        train_end = int(total_pairs * train_ratio)
        val_end = train_end + int(total_pairs * val_ratio)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        splits = {
            'train': [image_pairs[i] for i in train_indices],
            'valid': [image_pairs[i] for i in val_indices],
            'test': [image_pairs[i] for i in test_indices]
        }
        
        print("Split distribution:")
        print(f"  Train: {len(train_indices):4d} ({len(train_indices)/total_pairs*100:.1f}%)")
        print(f"  Valid: {len(val_indices):4d} ({len(val_indices)/total_pairs*100:.1f}%)")
        print(f"  Test:  {len(test_indices):4d} ({len(test_indices)/total_pairs*100:.1f}%)")
        print()
        
        # Process each split
        for split_name, pairs in splits.items():
            if len(pairs) == 0:
                continue
            
            print(f"{'='*70}")
            print(f"Processing {split_name.upper()} split ({len(pairs)} images)")
            print(f"{'='*70}")
            
            # Create output directories
            output_blur = self.output_dir / split_name / 'blur'
            output_sharp = self.output_dir / split_name / 'sharp'
            output_blur.mkdir(parents=True, exist_ok=True)
            output_sharp.mkdir(parents=True, exist_ok=True)
            
            # Prepare processing tasks
            tasks = []
            for blur_src, sharp_src, filename in pairs:
                tasks.append({
                    'blur_src': blur_src,
                    'sharp_src': sharp_src,
                    'blur_dst': output_blur / filename,
                    'sharp_dst': output_sharp / filename,
                    'target_size': (new_w, new_h)
                })
            
            # Process in parallel
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self._process_pair, task) for task in tasks]
                
                with tqdm(total=len(tasks), desc=f"{split_name:5s}", unit="pair") as pbar:
                    for future in as_completed(futures):
                        try:
                            future.result()
                            pbar.update(1)
                        except Exception as e:
                            print(f"\n❌ Error: {e}")
                            pbar.update(1)
            
            print(f"✅ {split_name.upper()} complete!\n")
        
        self._print_final_statistics()
    
    def _process_pair(self, task):
        """Process a single blur/sharp pair"""
        # Load images
        blur_img = Image.open(task['blur_src']).convert('RGB')
        sharp_img = Image.open(task['sharp_src']).convert('RGB')
        
        # Downsample using high-quality Lanczos filter
        blur_small = blur_img.resize(task['target_size'], Image.LANCZOS)
        sharp_small = sharp_img.resize(task['target_size'], Image.LANCZOS)
        
        # Save with compression
        blur_small.save(task['blur_dst'], 'PNG', optimize=True)
        sharp_small.save(task['sharp_dst'], 'PNG', optimize=True)
    
    def _print_final_statistics(self):
        """Print final dataset statistics"""
        print(f"\n{'='*70}")
        print("FINAL STATISTICS")
        print(f"{'='*70}\n")
        
        total_pairs = 0
        total_size_mb = 0
        
        for split in ['train', 'valid', 'test']:
            split_dir = self.output_dir / split
            
            if not split_dir.exists():
                continue
            
            blur_dir = split_dir / 'blur'
            sharp_dir = split_dir / 'sharp'
            
            blur_count = len(list(blur_dir.glob('*.png')))
            sharp_count = len(list(sharp_dir.glob('*.png')))
            
            if blur_count > 0:
                # Get sample image size
                sample_img = list(blur_dir.glob('*.png'))[0]
                img = Image.open(sample_img)
                
                # Calculate disk usage
                blur_files = list(blur_dir.glob('*.png'))
                sharp_files = list(sharp_dir.glob('*.png'))
                size_mb = sum(f.stat().st_size for f in blur_files + sharp_files) / (1024**2)
                
                total_pairs += blur_count
                total_size_mb += size_mb
                
                print(f"{split.upper():5s} Split:")
                print(f"  Pairs:      {blur_count:4d}")
                print(f"  Image size: {img.size[0]}x{img.size[1]}")
                print(f"  Disk usage: {size_mb:6.1f} MB")
                
                # Sample filenames
                print(f"  Samples:")
                for fname in sorted([f.name for f in blur_files])[:3]:
                    print(f"    - {fname}")
                print()
        
        print(f"TOTAL:")
        print(f"  Pairs: {total_pairs}")
        print(f"  Size:  {total_size_mb:.1f} MB")
        print(f"{'='*70}\n")
    
    def verify_dataset(self):
        """Verify processed dataset integrity"""
        print(f"\n{'='*70}")
        print("VERIFYING DATASET INTEGRITY")
        print(f"{'='*70}\n")
        
        all_ok = True
        
        for split in ['train', 'valid', 'test']:
            split_dir = self.output_dir / split
            
            if not split_dir.exists():
                print(f"⚠️  {split} split not found!")
                continue
            
            blur_dir = split_dir / 'blur'
            sharp_dir = split_dir / 'sharp'
            
            blur_images = set([f.name for f in blur_dir.glob('*.png')])
            sharp_images = set([f.name for f in sharp_dir.glob('*.png')])
            
            missing_sharp = blur_images - sharp_images
            missing_blur = sharp_images - blur_images
            
            print(f"{split.upper():5s} Split:")
            print(f"  Blur:  {len(blur_images):4d}")
            print(f"  Sharp: {len(sharp_images):4d}")
            
            if missing_sharp or missing_blur:
                all_ok = False
                if missing_sharp:
                    print(f"  ❌ {len(missing_sharp)} blur images missing sharp pairs")
                if missing_blur:
                    print(f"  ❌ {len(missing_blur)} sharp images missing blur pairs")
            else:
                print(f"  ✅ All pairs matched!")
            
            print()
        
        if all_ok:
            print("✅ Dataset verification passed!")
        else:
            print("⚠️  Dataset has mismatched pairs")
        
        return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Kaggle GoPro dataset for faster training"
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to gopro_deblur folder (contains blur/images/ and sharp/images/)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='dataset/GOPRO_processed',
        help='Output directory for processed dataset'
    )
    parser.add_argument(
        '--downsample',
        type=int,
        default=4,
        choices=[1, 2, 4, 8],
        help='Downsample factor: 1 (no downsample), 2, 4, or 8'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training split ratio (default: 0.8)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation split ratio (default: 0.1)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Test split ratio (default: 0.1)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing dataset'
    )
    
    args = parser.parse_args()
    
    # Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"❌ Split ratios must sum to 1.0 (got {total_ratio})")
        return
    
    # Display configuration
    print(f"\n{'='*70}")
    print("KAGGLE GOPRO DATASET PREPROCESSING")
    print(f"{'='*70}")
    print(f"Source:      {args.source}")
    print(f"Output:      {args.output}")
    print(f"Downsample:  {args.downsample}x")
    print(f"Workers:     {args.workers}")
    print(f"Split:")
    print(f"  Train:     {args.train_ratio*100:.0f}%")
    print(f"  Valid:     {args.val_ratio*100:.0f}%")
    print(f"  Test:      {args.test_ratio*100:.0f}%")
    print(f"{'='*70}\n")
    
    preprocessor = KaggleGoPROPreprocessor(
        source_dir=args.source,
        output_dir=args.output,
        downsample_factor=args.downsample,
        num_workers=args.workers
    )
    
    if args.verify_only:
        preprocessor.verify_dataset()
        return
    
    # Process dataset
    preprocessor.process_dataset(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    # Verify integrity
    preprocessor.verify_dataset()
    
    # Final summary
    size_map = {
        1: "1280x720 (original)",
        2: "640x360",
        4: "320x176",
        8: "160x88"
    }
    
    print(f"\n{'='*70}")
    print("PREPROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nDataset ready at: {args.output}")
    print(f"\nResolution: {size_map.get(args.downsample, 'custom')}")
    print(f"Speedup: ~{args.downsample**2}x faster training")
    print(f"\nExpected structure:")
    print(f"  {args.output}/")
    print(f"    ├── train/")
    print(f"    │   ├── blur/  (~{int(1029*args.train_ratio)} images)")
    print(f"    │   └── sharp/")
    print(f"    ├── valid/")
    print(f"    │   ├── blur/  (~{int(1029*args.val_ratio)} images)")
    print(f"    │   └── sharp/")
    print(f"    └── test/")
    print(f"        ├── blur/  (~{int(1029*args.test_ratio)} images)")
    print(f"        └── sharp/")
    print(f"\nTo use with your training scripts:")
    print(f"  python main.py \\")
    print(f"    --data_dir {args.output} \\")
    print(f"    --model_name MIMO-UNet-Enhanced \\")
    print(f"    --batch_size {16 if args.downsample >= 4 else 8} \\")
    print(f"    --num_epoch 500")


if __name__ == '__main__':
    main()