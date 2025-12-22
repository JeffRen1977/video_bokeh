#!/usr/bin/env python3
"""
Chapter 19 Tutorial 3: Depth Estimation with MiDaS

This tutorial demonstrates depth estimation using the MiDaS model:
- Loading and using MiDaS model
- Understanding DPT (Dense Prediction Transformer) architecture
- Depth map interpretation and visualization
- Performance characteristics

Reference: Chapter 19, Section 19.2.1 - MiDaS (Mixed-data Depth Estimation)
"""

import cv2
import numpy as np
import torch
import time
from pathlib import Path

# Create output directory
output_dir = Path(__file__).parent / "results"
output_dir.mkdir(exist_ok=True)


def load_midas_model(device='cpu'):
    """
    Load MiDaS depth estimation model.
    
    This demonstrates Concept 1 from Chapter 19: MiDaS Model Architecture
    - Uses DPT-Large (Dense Prediction Transformer)
    - Pre-trained on mixed datasets for robustness
    - Transformer-based architecture with ResNet backbone
    
    Reference: Chapter 19, Section 19.2.1 - MiDaS Approach
    
    Args:
        device: 'cpu' or 'cuda'
        
    Returns:
        model: Loaded MiDaS model
        transform: Preprocessing transform
    """
    print("📥 Loading MiDaS model (DPT-Large)...")
    print("   This may take a moment on first run (downloading model)")
    
    # Load model from torch hub
    # DPT_Large is the recommended model for best quality
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    model.eval()
    
    # Load preprocessing transform
    transform = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transform.dpt_transform
    
    # Move to device
    if device == 'cuda' and torch.cuda.is_available():
        model = model.to(device)
        print("   ✅ Model loaded on GPU")
    else:
        print("   ✅ Model loaded on CPU")
    
    return model, transform


def preprocess_image_midas(image_bgr, transform):
    """
    Preprocess image for MiDaS model.
    
    MiDaS expects:
    - RGB format (not BGR)
    - Specific input size (handled by transform)
    - Normalized values
    
    Args:
        image_bgr: Input BGR image
        transform: MiDaS transform function
        
    Returns:
        input_tensor: Preprocessed tensor ready for model
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Apply transform (resizes, normalizes, etc.)
    input_tensor = transform(image_rgb)
    
    return input_tensor


def estimate_depth_midas(image_bgr, model, transform, device='cpu'):
    """
    Estimate depth map using MiDaS model.
    
    This demonstrates the complete depth estimation pipeline:
    1. Preprocess image
    2. Run model inference
    3. Post-process depth map
    4. Resize to original image size
    
    Reference: Chapter 19, Section 19.2.1 - MiDaS Depth Estimation
    
    Args:
        image_bgr: Input BGR image
        model: MiDaS model
        transform: Preprocessing transform
        device: 'cpu' or 'cuda'
        
    Returns:
        depth_map: Estimated depth map (numpy array)
        inference_time: Time taken for inference
    """
    print("⚡ Running MiDaS depth estimation...")
    
    # Preprocess
    input_tensor = preprocess_image_midas(image_bgr, transform)
    
    # Add batch dimension and move to device
    if device == 'cuda' and torch.cuda.is_available():
        input_tensor = input_tensor.to(device)
    
    # Run inference
    start_time = time.time()
    with torch.no_grad():
        prediction = model(input_tensor)
        
        # Interpolate to original image size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
    
    inference_time = time.time() - start_time
    
    print(f"   ✅ Depth estimation completed in {inference_time:.2f} seconds")
    print(f"   📊 Depth range: [{prediction.min():.3f}, {prediction.max():.3f}]")
    
    return prediction, inference_time


def visualize_midas_depth(depth_map, original_image):
    """
    Visualize MiDaS depth map with original image.
    
    Args:
        depth_map: MiDaS depth map
        original_image: Original BGR image
        
    Returns:
        visualization: Combined visualization
    """
    # Normalize depth for visualization
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    
    # Apply INFERNO colormap (as used in Chapter 19)
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    
    # Resize images to same size for side-by-side comparison
    h, w = original_image.shape[:2]
    max_h = 400
    if h > max_h:
        scale = max_h / h
        new_h, new_w = int(h * scale), int(w * scale)
        original_resized = cv2.resize(original_image, (new_w, new_h))
        depth_resized = cv2.resize(depth_colored, (new_w, new_h))
    else:
        original_resized = original_image
        depth_resized = depth_colored
    
    # Combine side by side
    combined = np.hstack([original_resized, depth_resized])
    
    # Add labels
    cv2.putText(combined, "Original", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "MiDaS Depth Map", (original_resized.shape[1] + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return combined, depth_colored


def analyze_depth_characteristics(depth_map):
    """
    Analyze characteristics of MiDaS depth map.
    
    This helps understand what MiDaS depth maps represent:
    - Relative depth (not metric)
    - Higher values = closer to camera
    - Lower values = farther from camera
    
    Reference: Chapter 19, Section 19.2.1 - Depth Map Interpretation
    
    Args:
        depth_map: MiDaS depth map
        
    Returns:
        analysis: Dictionary with statistics
    """
    analysis = {
        'min': depth_map.min(),
        'max': depth_map.max(),
        'mean': depth_map.mean(),
        'std': depth_map.std(),
        'percentiles': {
            '25th': np.percentile(depth_map, 25),
            '50th': np.percentile(depth_map, 50),
            '75th': np.percentile(depth_map, 75),
            '85th': np.percentile(depth_map, 85),
        }
    }
    
    return analysis


def main():
    """
    Main tutorial function demonstrating MiDaS depth estimation from Chapter 19.
    """
    print("=" * 70)
    print("Chapter 19 Tutorial 3: Depth Estimation with MiDaS")
    print("=" * 70)
    print("\nThis tutorial demonstrates:")
    print("1. Loading MiDaS DPT-Large model")
    print("2. Preprocessing images for MiDaS")
    print("3. Running depth estimation inference")
    print("4. Understanding MiDaS depth maps")
    print("\nReference: Chapter 19, Section 19.2.1\n")
    
    # Load test image
    image_path = Path(__file__).parent.parent / "data" / "portrait1.jpg"
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"📷 Loading image: {image_path.name}")
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
    
    print("\n" + "=" * 70)
    print("Step 1: Loading MiDaS Model")
    print("=" * 70)
    print("MiDaS (Mixed-data Depth Estimation):")
    print("- Architecture: DPT-Large (Dense Prediction Transformer)")
    print("- Backbone: ResNet + Transformer")
    print("- Training: Mixed datasets for robustness")
    print("- Output: Relative depth maps")
    print("- Typical speed: ~1.0s per image (CPU)")
    
    try:
        model, transform = load_midas_model(device='cpu')
    except Exception as e:
        print(f"❌ Failed to load MiDaS model: {e}")
        print("   Make sure you have internet connection for first-time download")
        return
    
    print("\n" + "=" * 70)
    print("Step 2: Depth Estimation")
    print("=" * 70)
    print("Running MiDaS inference on the image...")
    
    depth_map, inference_time = estimate_depth_midas(image, model, transform, device='cpu')
    
    print("\n" + "=" * 70)
    print("Step 3: Depth Map Analysis")
    print("=" * 70)
    
    analysis = analyze_depth_characteristics(depth_map)
    print(f"Depth Statistics:")
    print(f"  Min: {analysis['min']:.3f}")
    print(f"  Max: {analysis['max']:.3f}")
    print(f"  Mean: {analysis['mean']:.3f}")
    print(f"  Std: {analysis['std']:.3f}")
    print(f"\nPercentiles (useful for threshold selection):")
    print(f"  25th: {analysis['percentiles']['25th']:.3f}")
    print(f"  50th: {analysis['percentiles']['50th']:.3f}")
    print(f"  75th: {analysis['percentiles']['75th']:.3f}")
    print(f"  85th: {analysis['percentiles']['85th']:.3f} (used in adaptive strategy)")
    
    print("\n" + "=" * 70)
    print("Step 4: Visualization")
    print("=" * 70)
    print("Creating depth map visualization...")
    
    combined_viz, depth_colored = visualize_midas_depth(depth_map, image)
    
    # Save results
    combined_path = output_dir / "03_midas_depth_comparison.jpg"
    depth_path = output_dir / "03_midas_depth_map.jpg"
    
    cv2.imwrite(str(combined_path), combined_viz)
    cv2.imwrite(str(depth_path), depth_colored)
    
    print(f"✅ Comparison saved: {combined_path}")
    print(f"✅ Depth map saved: {depth_path}")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("✅ You've learned:")
    print("  1. How to load and use MiDaS DPT-Large model")
    print("  2. How MiDaS preprocesses images")
    print("  3. How to interpret MiDaS depth maps")
    print("  4. Performance characteristics of MiDaS")
    print(f"\n📊 Performance: {inference_time:.2f}s per image")
    print("\n📖 Key Points from Chapter 19:")
    print("  - MiDaS uses relative depth (not metric)")
    print("  - Higher values = closer to camera")
    print("  - Lower values = farther from camera")
    print("  - Robust across diverse scenes (mixed training)")
    print("\n📖 Next: Tutorial 4 - Depth Estimation with Depth-Anything (04_depth_estimation_depth_anything.py)")
    print("=" * 70)


if __name__ == "__main__":
    main()


