"""
GoPro Dataset 4x Downsampling Pipeline
Organizes dataset into blur/sharp pairs and downsamples for faster training
"""

import os
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np


class GoPROPreprocessor:
    """
    Preprocesses GoPro Large dataset:
    1. Organizes into flat blur/sharp structure
    2. Downsamples images by 4x (1280x720 -> 320x180)
    """
    
    def __init__(self, source_dir, output_dir, downsample_factor=4, num_workers=8):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.downsample_factor = downsample_factor
        self.num_workers = num_workers
        
    def organize_and_downsample(self, split='train'):
        """
        Main processing function
        
        Args:
            split: 'train' or 'test'
        """
        print(f"\n{'='*70}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*70}\n")
        
        # Setup paths
        source_split = self.source_dir / split
        output_split = self.output_dir / split
        
        # Create output directories
        (output_split / 'blur').mkdir(parents=True, exist_ok=True)
        (output_split / 'sharp').mkdir(parents=True, exist_ok=True)
        
        # Find all video sequences
        video_dirs = sorted([d for d in source_split.iterdir() if d.is_dir()])
        
        print(f"Found {len(video_dirs)} video sequences")
        print(f"Downsample factor: {self.downsample_factor}x")
        print(f"Workers: {self.num_workers}\n")
        
        # Collect all image pairs
        image_pairs = []
        for video_dir in video_dirs:
            blur_dir = video_dir / 'blur'
            sharp_dir = video_dir / 'sharp'
            
            if not blur_dir.exists() or not sharp_dir.exists():
                print(f"⚠️  Skipping {video_dir.name} (missing blur/sharp folders)")
                continue
            
            # Get all blur images
            blur_images = sorted(list(blur_dir.glob('*.png')))
            
            for blur_img in blur_images:
                sharp_img = sharp_dir / blur_img.name
                
                if sharp_img.exists():
                    # Create unique filename: GOPROXXX_frame.png
                    new_name = f"{video_dir.name}_{blur_img.name}"
                    
                    image_pairs.append({
                        'blur_src': blur_img,
                        'sharp_src': sharp_img,
                        'blur_dst': output_split / 'blur' / new_name,
                        'sharp_dst': output_split / 'sharp' / new_name
                    })
        
        print(f"Total image pairs: {len(image_pairs)}")
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._process_pair, pair) 
                for pair in image_pairs
            ]
            
            # Progress bar
            with tqdm(total=len(image_pairs), desc="Processing") as pbar:
                for future in as_completed(futures):
                    try:
                        future.result()
                        pbar.update(1)
                    except Exception as e:
                        print(f"\n❌ Error: {e}")
                        pbar.update(1)
        
        print(f"\n✅ {split.upper()} split complete!")
        self._print_statistics(output_split)
    
    def _process_pair(self, pair):
        """Process a single blur/sharp pair"""
        # Load images
        blur_img = Image.open(pair['blur_src']).convert('RGB')
        sharp_img = Image.open(pair['sharp_src']).convert('RGB')
        
        # Get original size
        orig_w, orig_h = blur_img.size
        
        # Calculate new size
        new_w = orig_w // self.downsample_factor
        new_h = orig_h // self.downsample_factor
        
        # Downsample using high-quality Lanczos filter
        blur_small = blur_img.resize((new_w, new_h), Image.LANCZOS)
        sharp_small = sharp_img.resize((new_w, new_h), Image.LANCZOS)
        
        # Save
        blur_small.save(pair['blur_dst'], 'PNG', optimize=True)
        sharp_small.save(pair['sharp_dst'], 'PNG', optimize=True)
    
    def _print_statistics(self, split_dir):
        """Print dataset statistics"""
        blur_count = len(list((split_dir / 'blur').glob('*.png')))
        sharp_count = len(list((split_dir / 'sharp').glob('*.png')))
        
        # Get sample image size
        sample_blur = list((split_dir / 'blur').glob('*.png'))[0]
        img = Image.open(sample_blur)
        
        print(f"\n{'='*70}")
        print("STATISTICS")
        print(f"{'='*70}")
        print(f"Blur images:   {blur_count}")
        print(f"Sharp images:  {sharp_count}")
        print(f"Image size:    {img.size[0]}x{img.size[1]}")
        print(f"{'='*70}\n")
    
    def verify_dataset(self):
        """Verify the processed dataset integrity"""
        print(f"\n{'='*70}")
        print("VERIFYING DATASET INTEGRITY")
        print(f"{'='*70}\n")
        
        for split in ['train', 'test']:
            split_dir = self.output_dir / split
            
            if not split_dir.exists():
                print(f"⚠️  {split} split not found!")
                continue
            
            blur_dir = split_dir / 'blur'
            sharp_dir = split_dir / 'sharp'
            
            blur_images = set([f.name for f in blur_dir.glob('*.png')])
            sharp_images = set([f.name for f in sharp_dir.glob('*.png')])
            
            # Check for mismatches
            missing_sharp = blur_images - sharp_images
            missing_blur = sharp_images - blur_images
            
            print(f"{split.upper()} Split:")
            print(f"  Blur images:  {len(blur_images)}")
            print(f"  Sharp images: {len(sharp_images)}")
            
            if missing_sharp:
                print(f"  ⚠️  {len(missing_sharp)} blur images missing sharp pairs")
            if missing_blur:
                print(f"  ⚠️  {len(missing_blur)} sharp images missing blur pairs")
            
            if not missing_sharp and not missing_blur:
                print(f"  ✅ All pairs matched!")
            
            print()
    
    def create_validation_split(self, val_ratio=0.1):
        """
        Create validation split from training data
        
        Args:
            val_ratio: Fraction of training data to use for validation
        """
        print(f"\n{'='*70}")
        print(f"CREATING VALIDATION SPLIT ({val_ratio*100:.0f}%)")
        print(f"{'='*70}\n")
        
        train_blur = self.output_dir / 'train' / 'blur'
        train_sharp = self.output_dir / 'train' / 'sharp'
        
        val_blur = self.output_dir / 'valid' / 'blur'
        val_sharp = self.output_dir / 'valid' / 'sharp'
        
        val_blur.mkdir(parents=True, exist_ok=True)
        val_sharp.mkdir(parents=True, exist_ok=True)
        
        # Get all training images
        train_images = sorted([f.name for f in train_blur.glob('*.png')])
        
        # Randomly select validation images
        np.random.seed(42)
        num_val = int(len(train_images) * val_ratio)
        val_images = np.random.choice(train_images, num_val, replace=False)
        
        print(f"Moving {num_val} image pairs to validation set...")
        
        # Move images
        for img_name in tqdm(val_images):
            # Move blur
            shutil.move(
                str(train_blur / img_name),
                str(val_blur / img_name)
            )
            # Move sharp
            shutil.move(
                str(train_sharp / img_name),
                str(val_sharp / img_name)
            )
        
        print(f"\n✅ Validation split created!")
        print(f"  Training:   {len(list(train_blur.glob('*.png')))} pairs")
        print(f"  Validation: {len(list(val_blur.glob('*.png')))} pairs")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess GoPro dataset for faster training"
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to GoPro_Large dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='dataset/GOPRO_4x_downsampled',
        help='Output directory for processed dataset'
    )
    parser.add_argument(
        '--downsample',
        type=int,
        default=4,
        help='Downsample factor (default: 4)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--create-val',
        action='store_true',
        help='Create validation split from training data'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation split ratio (default: 0.1)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing dataset'
    )
    
    args = parser.parse_args()
    
    preprocessor = GoPROPreprocessor(
        source_dir=args.source,
        output_dir=args.output,
        downsample_factor=args.downsample,
        num_workers=args.workers
    )
    
    if args.verify_only:
        preprocessor.verify_dataset()
        return
    
    # Process train and test splits
    preprocessor.organize_and_downsample('train')
    preprocessor.organize_and_downsample('test')
    
    # Create validation split if requested
    if args.create_val:
        preprocessor.create_validation_split(args.val_ratio)
    
    # Verify integrity
    preprocessor.verify_dataset()
    
    print(f"\n{'='*70}")
    print("PREPROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nDataset ready at: {args.output}")
    print(f"\nExpected structure:")
    print(f"  {args.output}/")
    print(f"    ├── train/")
    print(f"    │   ├── blur/")
    print(f"    │   └── sharp/")
    print(f"    ├── valid/  (if --create-val)")
    print(f"    │   ├── blur/")
    print(f"    │   └── sharp/")
    print(f"    └── test/")
    print(f"        ├── blur/")
    print(f"        └── sharp/")
    print(f"\nOriginal size: 1280x720")
    print(f"Downsampled:   {1280//args.downsample}x{720//args.downsample}")
    print(f"\nTo use with your training scripts, update data_dir to:")
    print(f"  --data_dir {args.output}")


if __name__ == '__main__':
    main()