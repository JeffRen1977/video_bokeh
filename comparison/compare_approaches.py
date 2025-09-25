#!/usr/bin/env python3
"""
Comprehensive Comparison of MiDaS vs Depth-Anything Approaches

This script compares both depth estimation approaches for portrait effects:
1. MiDaS approach
2. Depth-Anything approach

It processes the same images with both approaches and creates detailed comparisons.
"""

import cv2
import numpy as np
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from approaches.midas.midas_portrait import MiDaSPortraitProcessor
from approaches.depth_anything.depth_anything_portrait import DepthAnythingPortraitProcessor


class PortraitApproachComparator:
    """Comprehensive comparator for portrait effect approaches."""
    
    def __init__(self, device='cpu'):
        """
        Initialize comparator with both processors.
        
        Args:
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device
        self.midas_processor = None
        self.depth_anything_processor = None
        
        print("🎯 Initializing Portrait Approach Comparator...")
        self._initialize_processors()
        print("✅ Comparator initialized!")
    
    def _initialize_processors(self):
        """Initialize both portrait processors."""
        print("📥 Loading MiDaS processor...")
        self.midas_processor = MiDaSPortraitProcessor(device=self.device)
        
        print("📥 Loading Depth-Anything processor...")
        self.depth_anything_processor = DepthAnythingPortraitProcessor(device=self.device)
    
    def compare_single_image(self, image_path, output_dir="comparison/results"):
        """
        Compare both approaches on a single image.
        
        Args:
            image_path: Path to input image
            output_dir: Output directory for results
            
        Returns:
            Dictionary with comparison results
        """
        print(f"\n🎭 Comparing approaches for: {image_path}")
        
        img_name = Path(image_path).stem
        os.makedirs(output_dir, exist_ok=True)
        
        # Process with MiDaS
        print("\n📊 Processing with MiDaS...")
        midas_start = time.time()
        midas_results = self.midas_processor.process_image(
            image_path, 
            output_dir=f"{output_dir}/midas"
        )
        midas_total_time = time.time() - midas_start
        
        # Process with Depth-Anything
        print("\n📊 Processing with Depth-Anything...")
        depth_anything_start = time.time()
        depth_anything_results = self.depth_anything_processor.process_image(
            image_path, 
            output_dir=f"{output_dir}/depth_anything"
        )
        depth_anything_total_time = time.time() - depth_anything_start
        
        # Create comprehensive comparison
        self._create_comparison_grid(
            image_path, midas_results, depth_anything_results, 
            midas_total_time, depth_anything_total_time, output_dir
        )
        
        # Generate comparison report
        report = self._generate_comparison_report(
            img_name, midas_results, depth_anything_results,
            midas_total_time, depth_anything_total_time
        )
        
        return {
            'midas': midas_results,
            'depth_anything': depth_anything_results,
            'timing': {
                'midas': midas_total_time,
                'depth_anything': depth_anything_total_time
            },
            'report': report
        }
    
    def _create_comparison_grid(self, image_path, midas_results, depth_anything_results,
                              midas_time, depth_anything_time, output_dir):
        """Create comprehensive comparison grid."""
        
        # Load original image
        original = cv2.imread(image_path)
        
        # Resize if needed
        max_size = 400
        if max(original.shape[:2]) > max_size:
            scale = max_size / max(original.shape[:2])
            new_h, new_w = int(original.shape[0] * scale), int(original.shape[1] * scale)
            original = cv2.resize(original, (new_w, new_h))
        
        h, w = original.shape[:2]
        cell_h, cell_w = h // 2, w // 2
        
        # Create large comparison grid: 4 rows, 6 columns
        grid_rows, grid_cols = 4, 6
        comparison = np.zeros((grid_rows * cell_h, grid_cols * cell_w, 3), dtype=np.uint8)
        
        # Row 1: Original, MiDaS Conservative, MiDaS Face-Aware, MiDaS Adaptive, Depth-Anything Conservative, Depth-Anything Face-Aware
        comparison[:cell_h, :cell_w] = cv2.resize(original, (cell_w, cell_h))
        comparison[:cell_h, cell_w:2*cell_w] = cv2.resize(midas_results['conservative']['result'], (cell_w, cell_h))
        comparison[:cell_h, 2*cell_w:3*cell_w] = cv2.resize(midas_results['face_aware']['result'], (cell_w, cell_h))
        comparison[:cell_h, 3*cell_w:4*cell_w] = cv2.resize(midas_results['adaptive']['result'], (cell_w, cell_h))
        comparison[:cell_h, 4*cell_w:5*cell_w] = cv2.resize(depth_anything_results['conservative']['result'], (cell_w, cell_h))
        comparison[:cell_h, 5*cell_w:] = cv2.resize(depth_anything_results['face_aware']['result'], (cell_w, cell_h))
        
        # Row 2: Depth-Anything Adaptive, MiDaS Depth, MiDaS Mask, Depth-Anything Depth, Depth-Anything Mask, Timing Info
        comparison[cell_h:2*cell_h, :cell_w] = cv2.resize(depth_anything_results['adaptive']['result'], (cell_w, cell_h))
        comparison[cell_h:2*cell_h, cell_w:2*cell_w] = cv2.resize(midas_results['conservative']['depth_map'], (cell_w, cell_h))
        comparison[cell_h:2*cell_h, 2*cell_w:3*cell_w] = cv2.resize(midas_results['conservative']['mask'], (cell_w, cell_h))
        comparison[cell_h:2*cell_h, 3*cell_w:4*cell_w] = cv2.resize(depth_anything_results['conservative']['depth_map'], (cell_w, cell_h))
        comparison[cell_h:2*cell_h, 4*cell_w:5*cell_w] = cv2.resize(depth_anything_results['conservative']['mask'], (cell_w, cell_h))
        
        # Create timing info image
        timing_img = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        timing_text = f"MiDaS: {midas_time:.1f}s\nDepth-Anything: {depth_anything_time:.1f}s"
        y_pos = 30
        for line in timing_text.split('\n'):
            cv2.putText(timing_img, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += 30
        comparison[cell_h:2*cell_h, 5*cell_w:] = timing_img
        
        # Row 3: Face-Aware comparisons
        comparison[2*cell_h:3*cell_h, :cell_w] = cv2.resize(midas_results['face_aware']['result'], (cell_w, cell_h))
        comparison[2*cell_h:3*cell_h, cell_w:2*cell_w] = cv2.resize(depth_anything_results['face_aware']['result'], (cell_w, cell_h))
        
        # Row 4: Conservative comparisons
        comparison[3*cell_h:, :cell_w] = cv2.resize(midas_results['conservative']['result'], (cell_w, cell_h))
        comparison[3*cell_h:, cell_w:2*cell_w] = cv2.resize(depth_anything_results['conservative']['result'], (cell_w, cell_h))
        
        # Add labels
        labels = [
            "Original", "MiDaS Conservative", "MiDaS Face-Aware", "MiDaS Adaptive", 
            "Depth-Anything Conservative", "Depth-Anything Face-Aware",
            "Depth-Anything Adaptive", "MiDaS Depth", "MiDaS Mask", "Depth-Anything Depth", 
            "Depth-Anything Mask", "Timing",
            "MiDaS Face-Aware", "Depth-Anything Face-Aware", "", "", "", "",
            "MiDaS Conservative", "Depth-Anything Conservative", "", "", "", ""
        ]
        
        for i, label in enumerate(labels):
            if label:  # Only add non-empty labels
                row = i // 6
                col = i % 6
                y_pos = row * cell_h + 20
                x_pos = col * cell_w + 10
                cv2.putText(comparison, label, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Save comparison grid
        img_name = Path(image_path).stem
        cv2.imwrite(os.path.join(output_dir, f"{img_name}_comprehensive_comparison.jpg"), comparison)
        
        print(f"📊 Comprehensive comparison saved: {img_name}_comprehensive_comparison.jpg")
    
    def _generate_comparison_report(self, img_name, midas_results, depth_anything_results,
                                  midas_time, depth_anything_time):
        """Generate detailed comparison report."""
        
        report = {
            'image': img_name,
            'timing': {
                'midas_total': midas_time,
                'depth_anything_total': depth_anything_time,
                'speedup': depth_anything_time / midas_time if midas_time > 0 else 0
            },
            'strategies': {
                'midas': {},
                'depth_anything': {}
            }
        }
        
        # Analyze each strategy
        for strategy in ['conservative', 'face_aware', 'adaptive']:
            if strategy in midas_results:
                report['strategies']['midas'][strategy] = {
                    'processing_time': midas_results[strategy]['processing_time'],
                    'available': True
                }
            
            if strategy in depth_anything_results:
                report['strategies']['depth_anything'][strategy] = {
                    'processing_time': depth_anything_results[strategy]['processing_time'],
                    'available': True
                }
        
        return report
    
    def compare_multiple_images(self, image_dir="data", output_dir="comparison/results"):
        """
        Compare approaches on multiple images.
        
        Args:
            image_dir: Directory containing input images
            output_dir: Output directory for results
            
        Returns:
            Dictionary with all comparison results
        """
        print(f"\n🎯 Comparing approaches on all images in: {image_dir}")
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(image_dir).glob(f"*{ext}"))
            image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ No images found in {image_dir}")
            return {}
        
        print(f"📷 Found {len(image_files)} images to process")
        
        all_results = {}
        total_midas_time = 0
        total_depth_anything_time = 0
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\n📸 Processing image {i}/{len(image_files)}: {image_file.name}")
            
            try:
                results = self.compare_single_image(str(image_file), output_dir)
                all_results[image_file.name] = results
                
                total_midas_time += results['timing']['midas']
                total_depth_anything_time += results['timing']['depth_anything']
                
            except Exception as e:
                print(f"❌ Failed to process {image_file.name}: {e}")
                continue
        
        # Generate summary report
        summary = {
            'total_images': len(image_files),
            'processed_images': len(all_results),
            'total_timing': {
                'midas': total_midas_time,
                'depth_anything': total_depth_anything_time,
                'average_midas': total_midas_time / len(all_results) if all_results else 0,
                'average_depth_anything': total_depth_anything_time / len(all_results) if all_results else 0
            },
            'results': all_results
        }
        
        self._save_summary_report(summary, output_dir)
        
        return summary
    
    def _save_summary_report(self, summary, output_dir):
        """Save summary report to file."""
        
        report_content = f"""# Portrait Effect Approaches Comparison Report

## Summary
- **Total Images**: {summary['total_images']}
- **Successfully Processed**: {summary['processed_images']}

## Performance Comparison
- **MiDaS Total Time**: {summary['total_timing']['midas']:.2f} seconds
- **Depth-Anything Total Time**: {summary['total_timing']['depth_anything']:.2f} seconds
- **MiDaS Average Time**: {summary['total_timing']['average_midas']:.2f} seconds per image
- **Depth-Anything Average Time**: {summary['total_timing']['average_depth_anything']:.2f} seconds per image

## Speed Comparison
- **Speedup**: {summary['total_timing']['depth_anything'] / summary['total_timing']['midas']:.2f}x {'(Depth-Anything faster)' if summary['total_timing']['depth_anything'] < summary['total_timing']['midas'] else '(MiDaS faster)'}

## Individual Image Results
"""
        
        for img_name, results in summary['results'].items():
            report_content += f"\n### {img_name}\n"
            report_content += f"- **MiDaS Time**: {results['timing']['midas']:.2f}s\n"
            report_content += f"- **Depth-Anything Time**: {results['timing']['depth_anything']:.2f}s\n"
        
        # Save report
        report_path = os.path.join(output_dir, "comparison_report.md")
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"📊 Summary report saved: {report_path}")


def main():
    """Main function for comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare MiDaS vs Depth-Anything approaches')
    parser.add_argument('--image', type=str, help='Single image to compare')
    parser.add_argument('--directory', type=str, default='data', help='Directory with images to compare')
    parser.add_argument('--output', type=str, default='comparison/results', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    
    args = parser.parse_args()
    
    comparator = PortraitApproachComparator(device=args.device)
    
    if args.image:
        # Compare single image
        results = comparator.compare_single_image(args.image, args.output)
        print("\n✅ Single image comparison completed!")
        
    else:
        # Compare multiple images
        results = comparator.compare_multiple_images(args.directory, args.output)
        print("\n✅ Multiple images comparison completed!")
    
    print(f"\n📊 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
