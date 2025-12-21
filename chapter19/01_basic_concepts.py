#!/usr/bin/env python3
"""
Chapter 19 Tutorial 1: Basic Concepts of Depth-Aware Portrait Effects

This tutorial introduces the fundamental concepts from Chapter 19:
- Understanding depth maps
- Basic depth visualization
- Simple blurring techniques

Reference: Chapter 19, Section 19.1 - Introduction to Depth-Aware Portrait Effects
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Create output directory
output_dir = Path(__file__).parent / "results"
output_dir.mkdir(exist_ok=True)


def create_simple_depth_map(image_shape):
    """
    Create a simple synthetic depth map for demonstration.
    
    This demonstrates Concept 1 from Chapter 19: Understanding Depth Maps
    - Depth maps represent distance from camera
    - Values typically range from 0 (far/background) to 1 (near/foreground)
    - In portrait photography, faces should have high depth values
    
    Args:
        image_shape: (height, width) of the image
        
    Returns:
        depth_map: 2D array representing depth values
    """
    h, w = image_shape[:2]
    depth_map = np.zeros((h, w), dtype=np.float32)
    
    # Create a radial depth pattern (center is closer = higher depth)
    # This simulates a typical portrait where the subject is in the center
    center_y, center_x = h // 2, w // 2
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    for y in range(h):
        for x in range(w):
            # Distance from center
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            # Normalize distance (0 to 1)
            normalized_dist = dist / max_dist
            # Center has high depth (close to camera), edges have low depth (far)
            depth_map[y, x] = 1.0 - normalized_dist * 0.7  # Range: 0.3 to 1.0
    
    return depth_map


def visualize_depth_map(depth_map, title="Depth Map"):
    """
    Visualize depth map using colormap.
    
    This demonstrates how depth maps are visualized in Chapter 19.
    - INFERNO colormap: Red = close, Yellow = far
    - VIRIDIS colormap: Purple = close, Yellow = far
    
    Args:
        depth_map: 2D depth map array
        title: Title for the visualization
    """
    # Normalize to 0-255 for visualization
    depth_normalized = ((depth_map - depth_map.min()) / 
                       (depth_map.max() - depth_map.min() + 1e-8) * 255).astype(np.uint8)
    
    # Apply colormap (INFERNO: commonly used in Chapter 19)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
    
    return depth_colored


def apply_simple_blur(image, depth_map, threshold=0.5):
    """
    Apply simple blur based on depth map.
    
    This demonstrates Concept 2 from Chapter 19: Basic Blurring
    - Areas with low depth (background) should be blurred
    - Areas with high depth (foreground) should remain sharp
    - Threshold determines the separation point
    
    Args:
        image: Input BGR image
        depth_map: Depth map (same size as image)
        threshold: Depth threshold (below this = blur, above = sharp)
        
    Returns:
        result: Blurred image
        mask: Blur mask (white = blur, black = sharp)
    """
    # Normalize depth map to 0-1
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    
    # Create blur mask: blur areas with LOW depth (background)
    # This is the CORRECTED LOGIC from Chapter 19, Section 19.4
    blur_mask = (depth_normalized < threshold).astype(np.float32)
    
    # Smooth the mask for better blending
    blur_mask = cv2.GaussianBlur(blur_mask, (15, 15), 0)
    
    # Apply blur to the entire image
    blurred = cv2.GaussianBlur(image, (51, 51), 0)
    
    # Composite: sharp foreground + blurred background
    result = image.copy().astype(np.float32)
    for c in range(3):
        result[:, :, c] = (
            result[:, :, c] * (1 - blur_mask) +  # Keep sharp (foreground)
            blurred[:, :, c] * blur_mask          # Blur (background)
        )
    
    return result.astype(np.uint8), blur_mask


def main():
    """
    Main tutorial function demonstrating basic concepts from Chapter 19.
    """
    print("=" * 70)
    print("Chapter 19 Tutorial 1: Basic Concepts of Depth-Aware Portrait Effects")
    print("=" * 70)
    print("\nThis tutorial demonstrates:")
    print("1. Creating and understanding depth maps")
    print("2. Visualizing depth information")
    print("3. Applying basic blurring based on depth")
    print("\nReference: Chapter 19, Section 19.1\n")
    
    # Load a test image
    image_path = Path(__file__).parent.parent / "data" / "portrait1.jpg"
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        print("Please ensure you have test images in the data/ directory")
        return
    
    print(f"📷 Loading image: {image_path.name}")
    image = cv2.imread(str(image_path))
    if image is None:
        print("❌ Failed to load image")
        return
    
    # Resize if too large for tutorial
    max_size = 600
    if max(image.shape[:2]) > max_size:
        scale = max_size / max(image.shape[:2])
        new_h, new_w = int(image.shape[0] * scale), int(image.shape[1] * scale)
        image = cv2.resize(image, (new_w, new_h))
        print(f"📏 Resized to: {new_h}x{new_w}")
    
    print("\n" + "=" * 70)
    print("Step 1: Creating Synthetic Depth Map")
    print("=" * 70)
    print("Creating a simple depth map that simulates portrait photography:")
    print("- Center region (subject) has HIGH depth (close to camera)")
    print("- Edge regions (background) have LOW depth (far from camera)")
    
    depth_map = create_simple_depth_map(image.shape)
    print(f"✅ Depth map created: shape={depth_map.shape}, range=[{depth_map.min():.3f}, {depth_map.max():.3f}]")
    
    print("\n" + "=" * 70)
    print("Step 2: Visualizing Depth Map")
    print("=" * 70)
    print("Converting depth map to colored visualization using INFERNO colormap")
    
    depth_visualization = visualize_depth_map(depth_map)
    depth_path = output_dir / "01_depth_map_visualization.jpg"
    cv2.imwrite(str(depth_path), depth_visualization)
    print(f"✅ Depth visualization saved: {depth_path}")
    
    print("\n" + "=" * 70)
    print("Step 3: Applying Depth-Based Blurring")
    print("=" * 70)
    print("Applying blur based on depth map:")
    print("- Areas with LOW depth (background) → BLURRED")
    print("- Areas with HIGH depth (foreground) → SHARP")
    print("- Threshold: 0.5 (adjustable)")
    
    result, blur_mask = apply_simple_blur(image, depth_map, threshold=0.5)
    
    # Save results
    result_path = output_dir / "01_portrait_effect.jpg"
    mask_path = output_dir / "01_blur_mask.jpg"
    
    cv2.imwrite(str(result_path), result)
    cv2.imwrite(str(mask_path), (blur_mask * 255).astype(np.uint8))
    
    print(f"✅ Portrait effect saved: {result_path}")
    print(f"✅ Blur mask saved: {mask_path}")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("✅ You've learned:")
    print("  1. How depth maps represent distance from camera")
    print("  2. How to visualize depth information")
    print("  3. How to apply blurring based on depth values")
    print("\n📖 Next: Tutorial 2 - Face Detection (02_face_detection.py)")
    print("=" * 70)


if __name__ == "__main__":
    main()

