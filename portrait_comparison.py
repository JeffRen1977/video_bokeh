#!/usr/bin/env python3
"""
Portrait Effect Comparison - Main Orchestrator

This script provides a unified interface to compare MiDaS and Depth-Anything approaches
for portrait effects with proper face sharpening.

Usage:
    python portrait_comparison.py --help
    python portrait_comparison.py --single data/portrait1.jpg
    python portrait_comparison.py --all
    python portrait_comparison.py --compare
"""

import argparse
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from approaches.midas.midas_portrait import MiDaSPortraitProcessor
from approaches.depth_anything.depth_anything_portrait import DepthAnythingPortraitProcessor
from comparison.compare_approaches import PortraitApproachComparator


def run_single_image(image_path, approach='both', output_dir='results'):
    """
    Process a single image with specified approach.
    
    Args:
        image_path: Path to input image
        approach: 'midas', 'depth_anything', or 'both'
        output_dir: Output directory
    """
    print(f"🎭 Processing single image: {image_path}")
    
    if approach in ['midas', 'both']:
        print("\n📊 Processing with MiDaS...")
        midas_processor = MiDaSPortraitProcessor()
        midas_results = midas_processor.process_image(image_path, f"{output_dir}/midas")
        print("✅ MiDaS processing completed!")
    
    if approach in ['depth_anything', 'both']:
        print("\n📊 Processing with Depth-Anything...")
        depth_anything_processor = DepthAnythingPortraitProcessor()
        depth_anything_results = depth_anything_processor.process_image(image_path, f"{output_dir}/depth_anything")
        print("✅ Depth-Anything processing completed!")
    
    print(f"💾 Results saved to: {output_dir}/")


def run_all_images(image_dir='data', approach='both', output_dir='results'):
    """
    Process all images in a directory with specified approach.
    
    Args:
        image_dir: Directory containing input images
        approach: 'midas', 'depth_anything', or 'both'
        output_dir: Output directory
    """
    print(f"🎭 Processing all images in: {image_dir}")
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f"*{ext}"))
        image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"📷 Found {len(image_files)} images to process")
    
    if approach in ['midas', 'both']:
        print("\n📊 Processing all images with MiDaS...")
        midas_processor = MiDaSPortraitProcessor()
        for i, image_file in enumerate(image_files, 1):
            print(f"  Processing {i}/{len(image_files)}: {image_file.name}")
            try:
                midas_processor.process_image(str(image_file), f"{output_dir}/midas")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
        print("✅ MiDaS processing completed!")
    
    if approach in ['depth_anything', 'both']:
        print("\n📊 Processing all images with Depth-Anything...")
        depth_anything_processor = DepthAnythingPortraitProcessor()
        for i, image_file in enumerate(image_files, 1):
            print(f"  Processing {i}/{len(image_files)}: {image_file.name}")
            try:
                depth_anything_processor.process_image(str(image_file), f"{output_dir}/depth_anything")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
        print("✅ Depth-Anything processing completed!")
    
    print(f"💾 Results saved to: {output_dir}/")


def run_comparison(image_dir='data', output_dir='comparison/results'):
    """
    Run comprehensive comparison between both approaches.
    
    Args:
        image_dir: Directory containing input images
        output_dir: Output directory for comparison results
    """
    print("🎯 Running comprehensive comparison...")
    
    comparator = PortraitApproachComparator()
    results = comparator.compare_multiple_images(image_dir, output_dir)
    
    print("✅ Comparison completed!")
    print(f"📊 Results saved to: {output_dir}/")
    
    return results


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Portrait Effect Comparison Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image with both approaches
  python portrait_comparison.py --single data/portrait1.jpg
  
  # Process single image with MiDaS only
  python portrait_comparison.py --single data/portrait1.jpg --approach midas
  
  # Process all images in data directory
  python portrait_comparison.py --all
  
  # Run comprehensive comparison
  python portrait_comparison.py --compare
  
  # Process all images with Depth-Anything only
  python portrait_comparison.py --all --approach depth_anything
        """
    )
    
    # Main action arguments
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--single', type=str, metavar='IMAGE_PATH',
                             help='Process a single image')
    action_group.add_argument('--all', action='store_true',
                             help='Process all images in data directory')
    action_group.add_argument('--compare', action='store_true',
                             help='Run comprehensive comparison between approaches')
    
    # Optional arguments
    parser.add_argument('--approach', type=str, choices=['midas', 'depth_anything', 'both'],
                       default='both', help='Which approach to use (default: both)')
    parser.add_argument('--input-dir', type=str, default='data',
                       help='Input directory for images (default: data)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results (default: results)')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu',
                       help='Device to use for processing (default: cpu)')
    
    args = parser.parse_args()
    
    print("🎯 Portrait Effect Comparison Tool")
    print("=" * 50)
    print(f"📁 Input directory: {args.input_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"💻 Device: {args.device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if args.single:
            # Process single image
            if not os.path.exists(args.single):
                print(f"❌ Image not found: {args.single}")
                return
            
            run_single_image(args.single, args.approach, args.output_dir)
            
        elif args.all:
            # Process all images
            if not os.path.exists(args.input_dir):
                print(f"❌ Input directory not found: {args.input_dir}")
                return
            
            run_all_images(args.input_dir, args.approach, args.output_dir)
            
        elif args.compare:
            # Run comparison
            if not os.path.exists(args.input_dir):
                print(f"❌ Input directory not found: {args.input_dir}")
                return
            
            run_comparison(args.input_dir, args.output_dir)
    
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    print("\n🎉 Processing completed successfully!")


if __name__ == "__main__":
    main()
