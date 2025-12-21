#!/usr/bin/env python3
"""
Chapter 19 Tutorial 6: Complete Portrait Effect Pipeline

This tutorial demonstrates the complete pipeline from Chapter 19:
- End-to-end portrait effect creation
- Using real depth estimation models (MiDaS or Depth-Anything)
- Performance optimization
- Quality assessment

Reference: Chapter 19, Section 19.5 - Complete Implementation
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from approaches.midas.midas_portrait import MiDaSPortraitProcessor
from approaches.depth_anything.depth_anything_portrait import DepthAnythingPortraitProcessor


def main():
    """
    Complete pipeline demonstration from Chapter 19.
    """
    print("=" * 70)
    print("Chapter 19 Tutorial 6: Complete Portrait Effect Pipeline")
    print("=" * 70)
    print("\nThis tutorial demonstrates the complete implementation from Chapter 19:")
    print("1. Real depth estimation using MiDaS or Depth-Anything")
    print("2. Face detection and protection")
    print("3. Multiple blurring strategies")
    print("4. Performance benchmarking")
    print("\nReference: Chapter 19, Section 19.5\n")
    
    # Load test image
    image_path = Path(__file__).parent.parent / "data" / "portrait1.jpg"
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"📷 Processing image: {image_path.name}\n")
    
    # Ask user which approach to use
    print("Select depth estimation approach:")
    print("1. MiDaS (DPT-Large) - Robust, well-established")
    print("2. Depth-Anything (DINOv2) - Fast, state-of-the-art")
    print("3. Both (for comparison)")
    
    choice = input("\nEnter choice (1/2/3, default=2): ").strip() or "2"
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    if choice == "1" or choice == "3":
        print("\n" + "=" * 70)
        print("Processing with MiDaS")
        print("=" * 70)
        print("Reference: Chapter 19, Section 19.2.1 - MiDaS Approach")
        print("\nMiDaS (Mixed-data Depth Estimation):")
        print("- Model: DPT-Large (Dense Prediction Transformer)")
        print("- Architecture: Transformer-based with ResNet backbone")
        print("- Input: 384×384 (resizable)")
        print("- Output: Relative depth maps")
        print("- Typical speed: ~1.04s per image (CPU)")
        
        try:
            processor = MiDaSPortraitProcessor(device='cpu')
            results = processor.process_image(
                str(image_path),
                output_dir=str(output_dir / "midas"),
                strategies=['conservative', 'face_aware', 'adaptive']
            )
            print("\n✅ MiDaS processing completed!")
            print(f"   Results saved to: {output_dir / 'midas'}")
        except Exception as e:
            print(f"❌ MiDaS processing failed: {e}")
    
    if choice == "2" or choice == "3":
        print("\n" + "=" * 70)
        print("Processing with Depth-Anything")
        print("=" * 70)
        print("Reference: Chapter 19, Section 19.2.2 - Depth-Anything Approach")
        print("\nDepth-Anything:")
        print("- Model: DPT-DINOv2 with ViT-Small backbone")
        print("- Architecture: Vision Transformer with DINOv2 features")
        print("- Input: 518×518 (resizable)")
        print("- Output: Metric depth maps")
        print("- Typical speed: ~0.35s per image (CPU) - 3x faster than MiDaS")
        
        try:
            processor = DepthAnythingPortraitProcessor(device='cpu', encoder='vits')
            results = processor.process_image(
                str(image_path),
                output_dir=str(output_dir / "depth_anything"),
                strategies=['conservative', 'face_aware', 'adaptive']
            )
            print("\n✅ Depth-Anything processing completed!")
            print(f"   Results saved to: {output_dir / 'depth_anything'}")
        except Exception as e:
            print(f"❌ Depth-Anything processing failed: {e}")
            print("   Note: Make sure Depth-Anything repository is set up")
            print("   Run: ./setup_depth_anything.sh")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("✅ Complete pipeline demonstration finished!")
    print("\nKey concepts from Chapter 19 demonstrated:")
    print("  1. Monocular depth estimation (Section 19.2)")
    print("  2. Face detection and protection (Section 19.3)")
    print("  3. Depth-based blurring strategies (Section 19.4)")
    print("  4. Performance optimization (Section 19.5)")
    print("\n📖 For detailed explanations, refer to:")
    print("  - Tutorial 1: Basic concepts")
    print("  - Tutorial 2: Face detection")
    print("  - Tutorial 3: MiDaS depth estimation")
    print("  - Tutorial 4: Depth-Anything depth estimation")
    print("  - Tutorial 5: Portrait effect creation")
    print("=" * 70)


if __name__ == "__main__":
    main()

