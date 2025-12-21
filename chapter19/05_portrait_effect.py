#!/usr/bin/env python3
"""
Chapter 19 Tutorial 5: Portrait Effect Creation

This tutorial demonstrates creating portrait effects by combining:
- Depth estimation (from Tutorial 3 or 4)
- Face detection (from Tutorial 2)
- Multiple blurring strategies
- Corrected depth logic

Reference: Chapter 19, Section 19.4 - Portrait Effect Implementation
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from approaches.midas.midas_portrait import MiDaSPortraitProcessor
from approaches.depth_anything.depth_anything_portrait import DepthAnythingPortraitProcessor

# Create output directory
output_dir = Path(__file__).parent / "results"
output_dir.mkdir(exist_ok=True)


def demonstrate_blurring_strategies(image_path, depth_approach='depth_anything'):
    """
    Demonstrate different blurring strategies from Chapter 19.
    
    This tutorial shows the three main strategies:
    1. Conservative: Gentle background blur (threshold=0.8)
    2. Face-Aware: Aggressive blur with face protection (threshold=0.2)
    3. Adaptive: Dynamic threshold based on depth distribution (85th percentile)
    
    Reference: Chapter 19, Section 19.4 - Blurring Strategies
    
    Args:
        image_path: Path to input image
        depth_approach: 'midas' or 'depth_anything'
    """
    print("=" * 70)
    print("Chapter 19 Tutorial 5: Portrait Effect Creation")
    print("=" * 70)
    print("\nThis tutorial demonstrates:")
    print("1. Combining depth estimation with blurring")
    print("2. Corrected depth logic implementation")
    print("3. Multiple blurring strategies")
    print("4. Face-aware processing")
    print("\nReference: Chapter 19, Section 19.4\n")
    
    print(f"📷 Loading image: {Path(image_path).name}")
    image = cv2.imread(str(image_path))
    if image is None:
        print("❌ Failed to load image")
        return
    
    # Resize if too large
    max_size = 800
    if max(image.shape[:2]) > max_size:
        scale = max_size / max(image.shape[:2])
        new_h, new_w = int(image.shape[0] * scale), int(image.shape[1] * scale)
        image = cv2.resize(image, (new_w, new_h))
        print(f"📏 Resized to: {new_h}x{new_w}")
    
    # Initialize processor
    print(f"\n📥 Initializing {depth_approach} processor...")
    if depth_approach == 'midas':
        processor = MiDaSPortraitProcessor(device='cpu')
    else:
        processor = DepthAnythingPortraitProcessor(device='cpu', encoder='vits')
    
    print("\n" + "=" * 70)
    print("Step 1: Depth Estimation")
    print("=" * 70)
    print("Estimating depth map...")
    
    depth_map = processor.estimate_depth(image)
    print(f"✅ Depth map estimated: shape={depth_map.shape}")
    
    print("\n" + "=" * 70)
    print("Step 2: Face Detection")
    print("=" * 70)
    print("Detecting faces for protection...")
    
    face_mask = processor._detect_faces(image)
    if face_mask is not None:
        print(f"✅ Faces detected and mask created")
        print(f"   Face coverage: {face_mask.sum() / face_mask.size * 100:.1f}% of image")
    else:
        print("⚠️  No faces detected")
    
    print("\n" + "=" * 70)
    print("Step 3: Portrait Effect Creation")
    print("=" * 70)
    print("Creating portrait effects with different strategies...")
    print("\nReference: Chapter 19, Section 19.4 - Blurring Strategies\n")
    
    strategies = {
        'conservative': {
            'threshold': 0.8,
            'description': 'Gentle background blur, keeps more foreground sharp',
            'use_case': 'Subtle portrait effect'
        },
        'face_aware': {
            'threshold': 0.2,
            'description': 'Aggressive blur with face protection',
            'use_case': 'Strong bokeh effect with face protection'
        },
        'adaptive': {
            'threshold': None,  # Will use 85th percentile
            'description': 'Dynamic threshold based on depth distribution',
            'use_case': 'Adaptive to image content'
        }
    }
    
    results = {}
    
    for strategy_name, strategy_info in strategies.items():
        print(f"\n--- {strategy_name.upper()} Strategy ---")
        print(f"Description: {strategy_info['description']}")
        print(f"Use Case: {strategy_info['use_case']}")
        
        if strategy_info['threshold'] is not None:
            print(f"Threshold: {strategy_info['threshold']}")
        else:
            print("Threshold: 85th percentile (adaptive)")
        
        # Create portrait effect
        result, bg_mask, depth_normalized = processor.create_portrait_effect(
            image, depth_map, face_mask, 
            strategy=strategy_name,
            threshold=strategy_info['threshold']
        )
        
        results[strategy_name] = {
            'result': result,
            'bg_mask': bg_mask,
            'depth_normalized': depth_normalized
        }
        
        print(f"✅ {strategy_name} portrait effect created")
    
    print("\n" + "=" * 70)
    print("Step 4: Saving Results")
    print("=" * 70)
    
    img_name = Path(image_path).stem
    
    for strategy_name, result_data in results.items():
        # Save portrait effect
        result_path = output_dir / f"05_{strategy_name}_portrait.jpg"
        cv2.imwrite(str(result_path), result_data['result'])
        print(f"✅ Saved: {result_path}")
        
        # Save blur mask
        mask_path = output_dir / f"05_{strategy_name}_mask.jpg"
        mask_viz = (result_data['bg_mask'] * 255).astype(np.uint8)
        mask_viz = cv2.applyColorMap(mask_viz, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(str(mask_path), mask_viz)
    
    # Save original and depth map
    original_path = output_dir / "05_original.jpg"
    depth_path = output_dir / "05_depth_map.jpg"
    
    cv2.imwrite(str(original_path), image)
    depth_viz = cv2.applyColorMap(
        (results['conservative']['depth_normalized'] * 255).astype(np.uint8),
        cv2.COLORMAP_INFERNO
    )
    cv2.imwrite(str(depth_path), depth_viz)
    
    print(f"✅ Saved: {original_path}")
    print(f"✅ Saved: {depth_path}")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("✅ You've learned:")
    print("  1. How to combine depth estimation with blurring")
    print("  2. The corrected depth logic (low = background, high = foreground)")
    print("  3. Three different blurring strategies:")
    print("     - Conservative: Gentle blur (threshold=0.8)")
    print("     - Face-Aware: Aggressive blur with face protection (threshold=0.2)")
    print("     - Adaptive: Dynamic threshold (85th percentile)")
    print("  4. How face detection protects facial regions")
    print("\n📖 Key Concepts from Chapter 19, Section 19.4:")
    print("  - Corrected depth logic ensures faces stay sharp")
    print("  - Multiple strategies for different use cases")
    print("  - Face protection adds robustness")
    print("  - Smooth blending creates natural effects")
    print("\n📖 Next: Tutorial 6 - Complete Pipeline (06_complete_pipeline.py)")
    print("=" * 70)


def main():
    """
    Main function for portrait effect tutorial.
    """
    # Load test image
    image_path = Path(__file__).parent.parent / "data" / "portrait1.jpg"
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    # Ask user which depth estimation to use
    print("Select depth estimation approach:")
    print("1. MiDaS (DPT-Large)")
    print("2. Depth-Anything (DINOv2) - Recommended")
    
    choice = input("\nEnter choice (1/2, default=2): ").strip() or "2"
    depth_approach = 'midas' if choice == "1" else 'depth_anything'
    
    demonstrate_blurring_strategies(image_path, depth_approach)


if __name__ == "__main__":
    main()

